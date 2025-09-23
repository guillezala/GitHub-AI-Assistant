import streamlit as st
import atexit

from utils.github_client import GitHubClient
from utils.chunking import Chunker
from utils.embeddings import Embedder
from utils.embeddings import PineconeVectorStore

from agents.rag import RAGAgent
from langchain_ollama  import ChatOllama

from agents.orchestrator import Orchestrator
from agents.github_agent import GitHubMCPAgent
from agents.github_exec_tool import GitHubExecTool

from utils.query_analysis import QueryAnalyzer
import asyncio
from utils.runner_async import AsyncRunner

def init_query_analyzer():
    """Inicializa el analizador de consultas"""
    if 'query_analyzer' not in st.session_state:
        llm = st.session_state.chat_llm
        st.session_state.query_analyzer = QueryAnalyzer(llm=llm, logger=streamlit_logger)
    return st.session_state.query_analyzer

st.set_page_config(page_title="GitHub README Processor", page_icon="🐙", layout="wide")

def streamlit_logger(msg: str):
    st.info(msg)

st.session_state.chat_llm = ChatOllama(
            model="qwen2.5:7b-instruct-q4_0",
            temperature=0.0,

            model_kwargs={
                "num_ctx": 2048,      
                "num_thread": 4,      
                "keep_alive": "30s"   
            }
        )

# 1) Async runner
if "runner" not in st.session_state:
    st.session_state.runner = AsyncRunner()

# 2) GitHub MCP 
if "github_tool" not in st.session_state:
    gh = GitHubMCPAgent()
    try:

        st.session_state.runner.run(gh.connect())
    except Exception as e:
        st.warning(f"Could not connect to MCP Server: {e}")

    allowed_tools = {
        "list_pull_requests", "list_releases", "list_issues", "get_file_contents", "get_pull_request", "get_issue", "get_release_by_tag"
    }
    executor = st.session_state.runner.run(
        gh.build_executor(allowed_tools=allowed_tools, model="qwen2.5:7b-instruct-q4_0", temperature=0.0, max_iterations=5)
    )
    st.session_state.gh_client = gh
    st.session_state.github_tool = GitHubExecTool(executor=executor)

# 3) RAG 
if "rag_tool" not in st.session_state:
    rag_tool = RAGAgent(vector_store=PineconeVectorStore(index_name="repo-text-embed-index"),
                        embedder=Embedder(),
                        llm=st.session_state.chat_llm)
    st.session_state.rag_tool = rag_tool

# 4) Orchestrator -> build executor 
if "orchestrator" not in st.session_state:
    judge_llm = st.session_state.chat_llm
    orchestrator = Orchestrator(
        tools=[(st.session_state.github_tool.name, st.session_state.github_tool),
                (st.session_state.rag_tool.name, st.session_state.rag_tool)
        ],
        llm=judge_llm,
        logger=streamlit_logger,
        timeout_s=6000.0
    )
    st.session_state.orchestrator = orchestrator
    st.session_state.orch_executor = st.session_state.runner.run(orchestrator.build_orchestrator())

def _cleanup():
    try:
        if "gh_client" in st.session_state:
            st.session_state.runner.run(st.session_state.gh_client.close())
    finally:
        if "runner" in st.session_state:
            st.session_state.runner.stop()

atexit.register(_cleanup)
st.title("🐙 GitHub AI Assistant")

# === SECTION 1 ===
st.header("📋 Process README")

col1, col2 = st.columns(2)
with col1:
    owner = st.text_input("Repository owner", "", key="owner_input")
with col2:
    repo = st.text_input("Name of the repository", "", key="repo_input")

if st.button("🚀 Process README", type="primary"):
    if not owner or not repo:
        st.warning("Please enter the owner and the repository name.")
    else:
        gh_client = GitHubClient()
        readme = gh_client.fetch_readme(owner, repo)
        if not readme:
            st.error("The README could not be downloaded.")
        else:
            st.success("✅ README downloaded successfully.")

            with st.spinner("📝 Dividing README into chunks..."):
                chunker = Chunker(max_tokens=800)
                chunks = chunker.chunk(readme, overlap=100)
            st.success(f"📄 README divided into {len(chunks)} chunks.")

            with st.spinner("🧠 Calculating embeddings..."):
                embedder = Embedder()
                embeddings = embedder.embed_chunks(chunks, normalize=True, return_with_text=True)
            st.success(f"✨ {len(embeddings)} embeddings were calculated.")

            try:
                document = "README"
                with st.spinner("💾 Registering embeddings in Pinecone..."):
                    vector_store = PineconeVectorStore(index_name="repo-text-embed-index")
                    vector_store.upsert_embeddings(embeddings, document, repo)
                st.success("🎉 Embeddings saved in Pinecone successfully.")
                
                st.session_state.readme_processed = True
                
            except Exception as e:
                st.error(f"❌ Error saving embeddings in Pinecone.: {e}")

# === SECTION 2 ===
st.header("💬 Query Repositories")


st.markdown("💡 You can make queries about any public GitHub repository. If you want to have access to the information contained in the README, process it before writing your questions.")

user_query = st.text_area(
    "✍️ **Write your question about any repository:**",
    value=st.session_state.get('user_query', ''),
    height=100,
    key="query_input"
)

send_query_button = st.button("🔍 Send query", type="primary")

if send_query_button:
    if not user_query.strip():
        st.warning("⚠️ Please, write a query.")
    else:
        with st.spinner("🔎 Analyzing query..."):
            """analyzer = init_query_analyzer()
            
            analysis = analyzer.analyze_query(user_query)
            
            is_relevant = analyzer.is_relevant_query(analysis, min_confidence)"""
            is_relevant = True
        
        if is_relevant:
            st.success("✅ Valid query. Processing...")
            
            with st.spinner("🤖 Searching for answer with the Orchestrator agent..."):
                try:
                    
                    respuesta = st.session_state.runner.run(st.session_state.orch_executor.ainvoke({"input": user_query}, include_run_info=True, return_intermediate_steps=False))

                    st.markdown("### 🎯 Answer:")
                    st.write(respuesta["output"] if isinstance(respuesta, dict) else respuesta)
                                        
                except Exception as e:
                    st.error(f"❌ Error processing the query: {str(e)}")
        
        #else:
            # 5. Manejar consulta irrelevante
            #st.write(analysis["reasoning"] + "\nPlease try with another query related to GitHub code.")


#Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🐙 GitHub README Processor | Query public repositories on GitHub | Powered by RAG + LLM
    </div>
    """, 
    unsafe_allow_html=True
)