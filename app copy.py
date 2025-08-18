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

def init_query_analyzer():
    """Inicializa el analizador de consultas"""
    if 'query_analyzer' not in st.session_state:
        llm = Ollama(model="llama3.2:1b", temperature=0)
        st.session_state.query_analyzer = QueryAnalyzer(llm=llm, logger=streamlit_logger)
    return st.session_state.query_analyzer

def show_query_analysis_sidebar(analysis):
    """Muestra el análisis de la consulta en la sidebar"""
    with st.sidebar:
        st.subheader("🔍 Análisis de Consulta")
        
        # Indicadores de relevancia
        col1, col2 = st.columns(2)
        with col1:
            if analysis['es_codigo_abierto']:
                st.success("✅ Código Abierto")
            else:
                st.error("❌ Código Abierto")
        
        with col2:
            if analysis['es_programacion']:
                st.success("✅ Programación")
            else:
                st.error("❌ Programación")
        
        # Confianza
        confidence = analysis['confianza']
        if confidence >= 0.7:
            st.success(f"🎯 Confianza: {confidence:.1%}")
        elif confidence >= 0.4:
            st.warning(f"🤔 Confianza: {confidence:.1%}")
        else:
            st.error(f"❌ Confianza: {confidence:.1%}")
        
        # Repositorio detectado
        if analysis.get('repositorio'):
            st.info(f"📁 Repo detectado: `{analysis['repositorio']}`")
        
        # Razonamiento (si existe)
        if analysis.get('razonamiento'):
            with st.expander("💭 Razonamiento"):
                st.write(analysis['razonamiento'])

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
    st.warning("⚠️ **Esta consulta podría no estar relacionada con el repositorio o programación.**")
    
    # Mostrar detalles del análisis
    with st.expander("Ver análisis detallado"):
        st.json(analysis)
    
    # Opciones para el usuario
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Procesar de todas formas", type="secondary"):
            return True
    
    with col2:
        if st.button("❌ Cancelar y reformular"):
            st.session_state.user_query = ""
            st.rerun()
    
    st.markdown("---")
    show_query_suggestions()
    return False

# Configuración de la página
st.set_page_config(
    page_title="GitHub README Processor",
    page_icon="🐙",
    layout="wide"
)

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
    
    # Modo debug
    debug_mode = st.checkbox("🔍 Mostrar análisis detallado")
    
    st.markdown("---")

# === SECCIÓN 1: PROCESAMIENTO DE README ===
st.header("📋 Paso 1: Procesar README")

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

# Botones de acción
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    send_query_button = st.button("🔍 Enviar Consulta", type="primary")

with col2:
    if st.button("📊 Solo Analizar"):
        if user_query.strip():
            analyzer = init_query_analyzer()
            with st.spinner("🔍 Analizando consulta..."):
                analysis = analyzer.analyze_query(user_query)
            show_query_analysis_sidebar(analysis)
        else:
            st.warning("Escribe una consulta primero.")

with col3:
    if st.button("🗑️ Limpiar"):
        st.session_state.user_query = ""
        st.rerun()

# Procesar consulta
if send_query_button:
    if not user_query.strip():
        st.warning("⚠️ Por favor, escribe una pregunta.")
    else:
        # Inicializar analizador
        analyzer = init_query_analyzer()
        
        # 1. Analizar la consulta
        with st.spinner("🔍 Analizando consulta..."):
            analysis = analyzer.analyze_query(user_query)
        
        # 2. Mostrar análisis en sidebar (si debug está activado)
        if debug_mode:
            show_query_analysis_sidebar(analysis)
        
        # 3. Verificar relevancia
        is_relevant = analyzer.is_relevant_query(user_query, min_confidence)
        
        if is_relevant:
            # 4. Procesar consulta relevante
            st.success("✅ Consulta válida. Procesando...")
            
            with st.spinner("🤖 Buscando respuesta con el agente Orchestrator..."):
                try:
                    # Inicializar componentes
                    embedder = Embedder()
                    vector_store = PineconeVectorStore(index_name="repo-text-embed-index")
                    llm = Ollama(model="llama3.2:1b", temperature=0)
                    rag_agent = RAGAgent(embedder=embedder, vector_store=vector_store, llm=llm)
                    github_agent = GitHubAgent(github_tool=GitHubTool(), llm=llm)

                    orchestrator = OrchestratorAgent(
                        agents=[("RAGAgent", rag_agent)],
                        llm=llm,
                        logger=streamlit_logger
                    )
                    
                    # Ejecutar consulta
                    respuesta = orchestrator.run(user_query)
                    
                    # Mostrar respuesta
                    st.markdown("### 🎯 Respuesta:")
                    st.write(respuesta)
                    
                    # Mostrar análisis en sidebar si está en modo debug
                    if debug_mode:
                        show_query_analysis_sidebar(analysis)
                    
                except Exception as e:
                    st.error(f"❌ Error al procesar la consulta: {str(e)}")
        
        else:
            # 5. Manejar consulta irrelevante
            force_process = handle_irrelevant_query(analysis)
            
            if force_process:
                st.info("🔄 Procesando en modo experimental...")
                
                with st.spinner("🤖 Procesando consulta..."):
                    try:
                        embedder = Embedder()
                        vector_store = PineconeVectorStore(index_name="repo-text-embed-index")
                        llm = Ollama(model="llama3.2:1b", temperature=0)
                        rag_agent = RAGAgent(embedder=embedder, vector_store=vector_store, llm=llm)

                        orchestrator = OrchestratorAgent(
                            agents=[("RAGAgent", rag_agent)],
                            llm=llm,
                            logger=streamlit_logger
                        )
                        
                        respuesta = orchestrator.run(user_query)
                        
                        st.warning("⚠️ **Resultado experimental:**")
                        st.write(respuesta)
                        
                    except Exception as e:
                        st.error(f"❌ Error al procesar la consulta: {str(e)}")

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