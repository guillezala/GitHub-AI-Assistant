import streamlit as st
from utils.github_client import GitHubClient
from utils.chunking import Chunker
from utils.embeddings import Embedder
from utils.embeddings import PineconeVectorStore
from agents.rag import RAGAgent
from langchain_community.llms import Ollama
from agents.orchestrator import OrchestratorAgent
from agents.github_agent import GitHubAgent, GitHubTool
from utils.query_analysis import QueryAnalyzer

def streamlit_logger(msg):
    st.info(msg)

st.title("Procesador de README de GitHub")

owner = st.text_input("Creador del repositorio (owner)", "")
repo = st.text_input("Nombre del repositorio", "")

if st.button("Procesar README"):
    if not owner or not repo:
        st.warning("Por favor, ingresa el creador y el nombre del repositorio.")
    else:
        # Descargar README
        gh_client = GitHubClient()
        readme = gh_client.fetch_readme(owner, repo)
        if not readme:
            st.error("No se pudo descargar el README.")
        else:
            st.success("README descargado correctamente.")

            # Chunking
            with st.spinner("Dividiendo README en chunks..."):
                chunker = Chunker(max_tokens=800)
                chunks = chunker.chunk(readme, overlap=100)
            st.write(f"README dividido en {len(chunks)} chunks.")

            # Embeddings
            with st.spinner("Calculando embeddings..."):
                embedder = Embedder()
                embeddings = embedder.embed_chunks(chunks, normalize=True, return_with_text=True)
            st.write(f"Se calcularon {len(embeddings)} embeddings.")

            try:
                document="README"
                with st.spinner("Guardando embeddings en Pinecone..."):
                    vector_store = PineconeVectorStore(index_name="repo-text-embed-index")
                    vector_store.upsert_embeddings(embeddings, document, repo)
                st.success("Embeddings guardados en Pinecone correctamente.")
            except Exception as e:
                st.error(f"Error al guardar los embeddings en Pinecone: {e}")

st.header("Consulta el README procesado")
user_query = st.text_input("Escribe tu pregunta sobre el repositorio")

if st.button("Enviar consulta"):
    if not user_query:
        st.warning("Por favor, escribe una pregunta.")
    else:
        with st.spinner("Buscando respuesta con el agente Orchestrator..."):
            embedder = Embedder()
            vector_store = PineconeVectorStore(index_name="repo-text-embed-index")
            llm = Ollama(model="llama3.2:1b", temperature=0)
            rag_agent = RAGAgent(embedder=embedder, vector_store=vector_store, llm=llm)
            github_agent = GitHubAgent(github_tool=GitHubTool(), llm=llm)  # Ajusta github_tool según tu implementación

            orchestrator = OrchestratorAgent(agents=[
                                                    ("RAGAgent", rag_agent)
                                                    #("GitHubAgent", github_agent),
                                                ], 
                                                llm=llm,
                                                logger=streamlit_logger)
            
            respuesta = orchestrator.run(user_query)
        st.write("Respuesta del agente Orchestrator:")
        st.write(respuesta)