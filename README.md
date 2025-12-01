# 🐙 GitHub AI Assistant

Un asistente de IA basado en **Streamlit** que te permite procesar **README** de repositorios de GitHub y consultar repositorios públicos utilizando **RAG** (Retrieval-Augmented Generation) y un agente de IA.

El sistema se integra con los siguientes frameworks:

| Framework | Propósito |
|-----------|-----------|
| **Ollama** | Inferencia LLM |
| **Langchain** | Agentes de IA |
| **GitHub MCP Server** | Datos de repositorio |
| **Pinecone** | Almacenamiento vectorial |

## 📋 Requisitos e Instalación

Antes de ejecutar la aplicación, asegúrate de tener lo siguiente configurado:

### 🔧 Ollama y Modelo LLM

Instala y descarga un modelo LLM localmente:

- **Instalar Ollama**: https://ollama.com/docs/installation
- **Descargar el modelo** usado por la app (ejemplo):
  
  ```bash
  # Bash/WSL/macOS:
  ollama pull qwen2.5:7b-instruct-q4_0
  ```
  
  ```powershell
  # Windows PowerShell:
  ollama pull qwen2.5:7b-instruct-q4_0
  ```

> ⚠️ **Nota**: Asegúrate de que el nombre del modelo coincida con el valor en `app.py` (ej: `qwen2.5:7b-instruct-q4_0`). El modelo puede ser grande y tardar en descargar.

### 🐳 Docker Desktop

Requerido para ejecutar el servidor GitHub MCP:

- **Descargar**: https://www.docker.com/products/docker-desktop
- **Ejecutar el servidor GitHub MCP** (requiere un GitHub PAT):
  
  ```bash
  # Bash/WSL/Linux/macOS:
  docker run --rm -i -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN ghcr.io/github/github-mcp-server --enable-command-logging --log-file /tmp/mcp.log stdio
  ```
  
  ```powershell
  # Windows PowerShell:
  docker run --rm -i -e GITHUB_PERSONAL_ACCESS_TOKEN=$env:GITHUB_TOKEN ghcr.io/github/github-mcp-server --enable-command-logging --log-file /tmp/mcp.log stdio
  ```

> 💡 Si usas `podman` u otro runtime, adapta el comando según sea necesario.

### 🔑 GitHub Personal Access Token (PAT)

Requerido para que el servidor MCP acceda a los datos del repositorio:

- **Crear PAT**: https://github.com/settings/tokens
- **Scopes recomendados**: `repo` (y otros según sea necesario)
- **Configurar como variable de entorno**:
  
  ```bash
  # Bash/WSL:
  export GITHUB_TOKEN="ghp_XXXXXXXXXXXXXXXXXXXX"
  ```
  
  ```powershell
  # PowerShell:
  $env:GITHUB_TOKEN = "ghp_XXXXXXXXXXXXXXXXXXXX"
  ```

### 📍 Pinecone API Key e Índice

Configura tu índice vectorial en Pinecone:

- **Crear cuenta**: https://www.pinecone.io/
- **Crear un índice** con dimensión `384`
- **Configurar variable de entorno**:
  
  ```bash
  # Bash/WSL:
  export PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
  ```
  
  ```powershell
  # PowerShell:
  $env:PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"
  ```

> 📌 Nombre de índice por defecto: `rag-index` | Dimensión: `384`

### 🐍 Python y Dependencias

Instala las dependencias del proyecto:

```bash
python -m venv .venv
.\.venv\Scripts\Activate      # Windows PowerShell
pip install -r requirements.txt
```

Ejecuta la app:

```bash
streamlit run app.py
```

## 🎨 Interfaz Streamlit

### 📊 Descripción General

La página principal de la app es un **panel de control de Streamlit** dividido en dos secciones principales:

| Sección | Descripción |
|---------|-------------|
| **Process README** | Obtiene e indexa un README de repositorio en el almacén vectorial (Pinecone) |
| **Query Repositories** | Escribe una consulta sobre un repositorio y obtén respuestas vía Orchestrator (RAG + Agente GitHub) |

---

### 📝 Process README

**Entradas:**
- Propietario del repositorio (campo de texto)
- Nombre del repositorio (campo de texto)

**Acciones:**
- Haz clic en *"Process README"* para obtener el README de GitHub
- La app dividirá el README en fragmentos, calculará embeddings y los cargará en Pinecone

**Feedback en UI:**
- Mensajes de éxito/advertencia/error mediante Streamlit
- Spinners que muestran progreso en fragmentación, embedding y carga a Pinecone

### 🔍 Query Repositories

**Entradas:**
- Área de texto libre para escribir una pregunta sobre un repositorio

**Acciones:**
- Haz clic en *"Send query"* para ejecutar tu pregunta a través del agente Orchestrator
- El Orchestrator puede ejecutar la herramienta RAG (búsqueda vectorial) y el agente GitHub (MCP) para obtener y combinar respuestas

**Salida:**
- La respuesta final (y cualquier mensaje de error) se muestra bajo *"Answer"*

### ⚙️ Comportamiento Adicional

- La app intenta conectarse al servidor GitHub MCP al iniciar; los errores de conexión aparecen como advertencias en la UI
- Utiliza un `AsyncRunner` para ejecutar tareas asincrónicas (conexión MCP, construcción de agentes, embeddings) sin bloquear la interfaz de Streamlit
- Para depuración, la app registra mensajes mediante el logger de Streamlit (aparecen como mensajes en la UI)
- El modelo LLM y herramientas de agente permitidas se configuran en `app.py` — cámbilos si es necesario para experimentos

## 🤖 Agentes y Herramientas

Esta sección describe los tres componentes principales del agente en el sistema (**RAGAgent**, **GitHubAgent** y **Orchestrator**), cómo funcionan, qué entradas/salidas esperan, y guía sobre cuándo usar cada uno.

### 🧠 RAGAgent

**Propósito:**
- Proporcionar respuestas de alto nivel y contextuales buscando en la base de datos vectorial (Pinecone) fragmentos relevantes del README y sintetizando resultados con el LLM

**Cuándo usarlo:**
- Ideal para preguntas sobre propósito del proyecto, arquitectura, configuración, ejemplos de uso, o cualquier tema donde el contexto derivado del README es suficiente

**Flujo de trabajo:**
- Acepta una consulta de texto libre
- Incrusta la consulta usando el embedder del proyecto (SentenceTransformer)
- Realiza una búsqueda de vecinos más cercanos en el almacén vectorial devolviendo los k fragmentos principales
- Ensambla fragmentos recuperados en un payload de contexto consciente de relevancia y llama al LLM para producir una respuesta concisa y fundamentada en contexto

**Entradas / Salidas:**
- **Entrada**: cadena de consulta simple
- **Salida**: cadena de respuesta sintetizada (opcionalmente con citas o fragmentos de chunks recuperados)

**Limitaciones:**
- ❌ No es apta para responder preguntas que requieren datos de repositorio en tiempo real o a nivel de archivo (contenidos de archivos, líneas específicas, historial git) — usa **GitHubAgent** para esas

### 🔗 GitHubAgent (GitHubMCPAgent + MCPTool)

**Propósito:**
- Proporcionar información granular y actualizada del repositorio interactuando con un servidor GitHub MCP local expuesto vía stdio

**Qué proporciona:**
- Acceso programático a un conjunto de herramientas expuestas por el servidor:
  - 📂 Listado de archivos
  - 📄 Lectura de contenido de archivos
  - 🔎 Búsqueda en repo
  - 📊 Diffs
  
  Las herramientas se descubren dinámicamente vía sesión MCP

**Cómo funciona:**
- Se conecta a un servidor MCP local (contenedor iniciado con Docker) usando un PAT disponible para el servidor
- Llama a `list_tools()` para enumerar capacidades MCP disponibles y envuelve cada herramienta como un MCPTool utilizable por agentes
- Cuando se invoca, un MCPTool formatea entrada de acción similar a JSON, llama a `session.call_tool(tool_name, args)` y analiza la salida de herramienta en una observación utilizable para el agente

**Entradas / Salidas:**
- **Entrada**: entrada de acción estructurada similar a JSON (campos específicos de la herramienta) o prompts legibles por humanos delegados por un agente orquestador
- **Salida**: observación de herramienta sin procesar (string/JSON), post-procesada por utilidades `process_tool_output` para mantener resultados consistentes para el LLM

**Notas de seguridad y operación:**
- 🔐 Requiere un `GITHUB_TOKEN` válido proporcionado al contenedor MCP
- ⏱️ Las llamadas son remotas (contenedor ↔ host) y pueden introducir latencia
- 💡 Prefiere RAG para búsquedas locales baratas y GitHubAgent para consultas autorizadas a nivel de archivo
- ⚠️ Las herramientas pueden exponer operaciones sensibles; valida entradas y restringe herramientas permitidas donde sea apropiado

### 🎼 Orchestrator

**Propósito:**
- Componer herramientas y LLMs en un único agente de estilo **ReAct** que decide cuándo llamar a herramientas RAG o GitHub y cómo combinar resultados en una respuesta final

**Responsabilidades principales:**
- Construir un `AgentExecutor` que registre herramientas disponibles (RAGAgent, MCPTools, GitHubExecTool)
- Implementar una plantilla de prompt estructurada que impulse el bucle: **Thought** → **Action** → **Action Input** → **Observation**
- Aplicar límites de iteración y timeouts para evitar invocación descontrolada de herramientas
- Enrutar resultados de herramientas de vuelta al LLM y producir la respuesta final mostrada al usuario

**Cómo elige herramientas:**
- El prompt instruye al LLM a:
  - ✅ Preferir **RAG** cuando README/contexto es suficiente
  - ✅ Llamar a herramientas **GitHub** cuando la pregunta requiere contenidos de archivo, líneas específicas, o estado de repositorio activo

**Integración asincrónica y runtime:**
- El Orchestrator puede combinar llamadas LLM sincrónicas y llamadas MCP asincrónicas
- La app usa un `AsyncRunner` con bucle de evento en background para ejecutar operaciones asincrónicas sin bloquear Streamlit
- La ejecución del agente captura y muestra errores de herramientas; el Orchestrator maneja reintentos, timeouts y fallbacks donde se configure

**Ejemplo de flujo:**

```
1️⃣ Consulta del usuario llega al Orchestrator
   ↓
2️⃣ Orchestrator invoca RAGAgent para recuperar contexto relevante del README
   ↓
3️⃣ LLM analiza contexto y decide que se necesita una verificación a nivel de archivo
   → Llama a herramienta GitHub MCP (vía GitHubAgent)
   ↓
4️⃣ Herramienta GitHub devuelve contenido de archivo
   ↓
5️⃣ Orchestrator envía contexto combinado al LLM para respuesta final
```

**Orientación:**
- ⚡ Ajusta las herramientas permitidas y el conteo de iteraciones dependiendo de consultas típicas (menos iteraciones = más rápido, más seguro)
- 💰 Usa RAGAgent para respuestas contextuales de **bajo costo**
- ✔️ Usa GitHubAgent cuando la **corrección** y **detalles actualizados** importan
