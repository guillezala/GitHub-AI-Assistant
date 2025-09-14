import streamlit as st
import atexit

from utils.github_client import GitHubClient
from utils.chunking import Chunker
from utils.embeddings import Embedder
from utils.embeddings import PineconeVectorStore

from agents.rag import RAGAgent
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama

from agents.orchestrator import Orchestrator
from agents.github_agent import GitHubMCPAgent
from agents.github_exec_tool import GitHubExecTool

from utils.query_analysis import QueryAnalyzer
import asyncio
from utils.runner_async import AsyncRunner

def streamlit_logger(msg):
    st.info(msg)

def get_chat_llm():
    if "chat_llm" not in st.session_state:
        st.session_state.chat_llm = ChatOllama(
            model="qwen2.5:7b-instruct-q4_0",
            temperature=0.0,
            # manda opciones a Ollama para reducir huella
            model_kwargs={
                "num_ctx": 2048,      # menos memoria
                "num_thread": 4,      # CPU estable
                "keep_alive": "30s"   # descarga del VRAM/RAM si está inactivo
            }
        )
    return st.session_state.chat_llm

def init_query_analyzer():
    """Inicializa el analizador de consultas"""
    if 'query_analyzer' not in st.session_state:
        llm = get_chat_llm()
        st.session_state.query_analyzer = QueryAnalyzer(llm=llm, logger=streamlit_logger)
    return st.session_state.query_analyzer


def show_query_suggestions():
    """Muestra sugerencias de consultas válidas"""
    st.info("💡 **Prueba con estas consultas de ejemplo:**")
    
    suggestions = [
        "¿Cómo instalar la librería numpy?",
        "¿Cuáles son las principales funcionalidades de React?",
        "¿Qué dependencias necesita el proyecto tensorflow?",
        "¿Cómo contribuir al repositorio de Django?",
        "¿Cuál es la licencia del proyecto pandas?",
        "¿Hay ejemplos de uso en el repositorio de scikit-learn?"
    ]
    
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        col = cols[i % 2]
        with col:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                st.session_state.user_query = suggestion
                st.rerun()

def handle_irrelevant_query(analysis):
    """Maneja consultas que no parecen relevantes"""
    st.write(analysis["razonamiento"])
    
    return False

st.set_page_config(page_title="GitHub README Processor", page_icon="🐙", layout="wide")

def streamlit_logger(msg: str):
    st.info(msg)



# 1) Un único runner vivo
if "runner" not in st.session_state:
    st.session_state.runner = AsyncRunner()

# 2) GitHub MCP -> conectar + build executor (dentro del runner)
if "github_tool" not in st.session_state:
    gh = GitHubMCPAgent()
    try:
        # Conexión MCP (stdio) en el loop del runner
        st.session_state.runner.run(gh.connect())
    except Exception as e:
        st.warning(f"No se pudo conectar al servidor MCP todavía: {e}")
    # Whitelist compacta de tools MCP
    allowed_tools = {
        "list_pull_requests",
    }
    executor = st.session_state.runner.run(
        gh.build_executor(allowed_tools=allowed_tools, model="qwen2.5:7b-instruct", temperature=0.0, max_iterations=3)
    )
    st.session_state.gh_client = gh
    st.session_state.github_tool = GitHubExecTool(executor=executor)

# 3) RAG (BaseTool síncrono)
"""if "rag_tool" not in st.session_state:
    rag_tool = RAGAgent()
    rag_tool.init_agent()  
    st.session_state.rag_tool = rag_tool"""

# 4) Orquestador (usa judge_llm chat; no llames .run en Streamlit)
if "orchestrator" not in st.session_state:
    judge_llm = get_chat_llm()
    orchestrator = Orchestrator(
        tools=[st.session_state.github_tool],
        llm=judge_llm,
        logger=streamlit_logger,
        timeout_s=300.0
    )
    st.session_state.orchestrator = orchestrator

# Cierre limpio al salir
def _cleanup():
    try:
        if "gh_client" in st.session_state:
            st.session_state.runner.run(st.session_state.gh_client.close())
    finally:
        if "runner" in st.session_state:
            st.session_state.runner.stop()

atexit.register(_cleanup)
st.title("🐙 Procesador de README de GitHub")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Control de confianza mínima
    min_confidence = st.slider(
        "Confianza mínima para consultas",
        min_value=0.1,
        max_value=1.0,
        value=0.2,  # Más permisivo para README específicos
        step=0.1,
        help="Nivel mínimo de confianza para procesar automáticamente la consulta"
    )
    
    st.markdown("---")

# === SECCIÓN 1: PROCESAMIENTO DE README ===
st.header("📋 Procesar README")

col1, col2 = st.columns(2)
with col1:
    owner = st.text_input("Creador del repositorio (owner)", "", key="owner_input")
with col2:
    repo = st.text_input("Nombre del repositorio", "", key="repo_input")

if st.button("🚀 Procesar README", type="primary"):
    if not owner or not repo:
        st.warning("Por favor, ingresa el creador y el nombre del repositorio.")
    else:
        # Descargar README
        gh_client = GitHubClient()
        readme = gh_client.fetch_readme(owner, repo)
        if not readme:
            st.error("No se pudo descargar el README.")
        else:
            st.success("✅ README descargado correctamente.")
            
            # Guardar info del repo en session state
            st.session_state.current_repo = f"{owner}/{repo}"

            # Chunking
            with st.spinner("📝 Dividiendo README en chunks..."):
                chunker = Chunker(max_tokens=800)
                chunks = chunker.chunk(readme, overlap=100)
            st.success(f"📄 README dividido en {len(chunks)} chunks.")

            # Embeddings
            with st.spinner("🧠 Calculando embeddings..."):
                embedder = Embedder()
                embeddings = embedder.embed_chunks(chunks, normalize=True, return_with_text=True)
            st.success(f"✨ Se calcularon {len(embeddings)} embeddings.")

            try:
                document = "README"
                with st.spinner("💾 Guardando embeddings en Pinecone..."):
                    vector_store = PineconeVectorStore(index_name="repo-text-embed-index")
                    vector_store.upsert_embeddings(embeddings, document, repo)
                st.success("🎉 Embeddings guardados en Pinecone correctamente.")
                
                # Marcar como procesado
                st.session_state.readme_processed = True
                
            except Exception as e:
                st.error(f"❌ Error al guardar los embeddings en Pinecone: {e}")

# === SECCIÓN 2: CONSULTAS ===
st.header("💬 Consultar Repositorios")

# Info sobre repositorios disponibles
if st.session_state.get('readme_processed', False):
    st.info(f"📁 Último README procesado: `{st.session_state.get('current_repo', 'Repositorio')}`")

st.markdown("💡 Puedes consultar cualquier repositorio que haya sido procesado previamente en la base de vectores.")

# Input de consulta
user_query = st.text_area(
    "✍️ **Escribe tu pregunta sobre cualquier repositorio:**",
    value=st.session_state.get('user_query', ''),
    height=100,
    placeholder="Ej: ¿Cómo instalar numpy? ¿Cuáles son las funcionalidades de tensorflow/tensorflow?",
    key="query_input"
)

# Boton de acción
send_query_button = st.button("🔍 Enviar Consulta", type="primary")

# Procesar consulta
if send_query_button:
    if not user_query.strip():
        st.warning("⚠️ Por favor, escribe una pregunta.")
    else:
        # Inicializar analizador
        with st.spinner("🔎 Analizando consulta..."):
            analyzer = init_query_analyzer()
            
            analysis = analyzer.analyze_query(user_query)
            
            is_relevant = analyzer.is_relevant_query(analysis, min_confidence)
        
        if is_relevant:
            # 4. Procesar consulta relevante
            st.success("✅ Consulta válida. Procesando...")
            
            with st.spinner("🤖 Buscando respuesta con el agente Orchestrator..."):
                try:
                    # Ejecutar consulta
                    respuesta = st.session_state.runner.run(st.session_state.github_tool._arun(user_query))
                    
                    # Mostrar respuesta
                    st.markdown("### 🎯 Respuesta:")
                    st.write(respuesta)
                    
                    
                except Exception as e:
                    st.error(f"❌ Error al procesar la consulta: {str(e)}")
        
        else:
            # 5. Manejar consulta irrelevante
            st.write(analysis["razonamiento"])

# Mostrar sugerencias si no hay consulta
if not user_query.strip():
    st.markdown("---")
    show_query_suggestions()

# Footer con información
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🐙 GitHub README Processor | Consulta repositorios procesados previamente | Powered by RAG + LLM
    </div>
    """, 
    unsafe_allow_html=True
)