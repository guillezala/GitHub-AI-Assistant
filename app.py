import streamlit as st
from utils.github_client import GitHubClient
from utils.chunking import Chunker
from utils.embeddings import Embedder
from utils.embeddings import PineconeVectorStore

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
                chunks = chunker.chunk(readme)
            st.write(f"README dividido en {len(chunks)} chunks.")

            # Embeddings
            with st.spinner("Calculando embeddings..."):
                embedder = Embedder()
                embeddings = embedder.embed_chunks(chunks, normalize=True)
            st.write(f"Se calcularon {len(embeddings)} embeddings.")
            st.write(embeddings[0] if embeddings else "No se pudo calcular el embedding.")

            try:
                with st.spinner("Guardando embeddings en Pinecone..."):
                    vector_store = PineconeVectorStore(index_name="repo_text_embed_index")
                    vector_store.upsert_embeddings(embeddings)
                st.success("Embeddings guardados en Pinecone correctamente.")
            except Exception as e:
                st.error(f"Error al guardar los embeddings en Pinecone: {e}")