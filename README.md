# 🐙 GitHub AI Assistant

A **Streamlit**-based AI assistant that allows you to process GitHub repository **README** files and query public repositories using **RAG** (Retrieval-Augmented Generation) and an AI agent.

The system integrates with the following frameworks:

| Framework | Purpose |
|-----------|---------|
| **Ollama** | LLM Inference |
| **Langchain** | AI Agents |
| **GitHub MCP Server** | Repository Data |
| **Pinecone** | Vector Storage |

## 📋 Requirements and Installation

Before running the application, make sure you have the following configured:

### 🔧 Ollama and LLM Model

Install and download an LLM model locally:

- **Install Ollama**: https://ollama.com/docs/installation
- **Download the model** used by the app (example):
  
  ```powershell
  # Windows PowerShell:
  ollama pull qwen2.5:7b-instruct-q4_0
  ```

> ⚠️ **Note**: Make sure the model name matches the value in `app.py` (e.g., `qwen2.5:7b-instruct-q4_0`). The model can be large and take time to download.

### 🐳 Docker Desktop

Required to run the GitHub MCP server:

- **Download**: https://www.docker.com/products/docker-desktop
- **Run the GitHub MCP server** (requires a GitHub PAT):
  
  ```bash
  # Bash/WSL/Linux/macOS:
  docker run --rm -i -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN ghcr.io/github/github-mcp-server --enable-command-logging --log-file /tmp/mcp.log stdio
  ```
  
  ```powershell
  # Windows PowerShell:
  docker run --rm -i -e GITHUB_PERSONAL_ACCESS_TOKEN=$env:GITHUB_TOKEN ghcr.io/github/github-mcp-server --enable-command-logging --log-file /tmp/mcp.log stdio
  ```

> 💡 If you use `podman` or another runtime, adapt the command accordingly.

### 🔑 GitHub Personal Access Token (PAT)

Required for the MCP server to access repository data:

- **Create PAT**: https://github.com/settings/tokens
- **Recommended scopes**: `repo` (and others as needed)
- **Configure as environment variable**:
  
  ```bash
  # Bash/WSL:
  export GITHUB_TOKEN="ghp_XXXXXXXXXXXXXXXXXXXX"
  ```
  
  ```powershell
  # PowerShell:
  $env:GITHUB_TOKEN = "ghp_XXXXXXXXXXXXXXXXXXXX"
  ```

### 📍 Pinecone API Key and Index

Configure your vector index in Pinecone:

- **Create account**: https://www.pinecone.io/
- **Create an index** with dimension `384`
- **Configure environment variable**:
  
  ```bash
  # Bash/WSL:
  export PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
  ```
  
  ```powershell
  # PowerShell:
  $env:PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"
  ```

> 📌 Default index name: `rag-index` | Dimension: `384`

###  Python and Dependencies

Install project dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\Activate      # Windows PowerShell
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## 🎨 Streamlit Interface

### 📊 Overview

The app's main page is a **Streamlit dashboard** divided into two main sections:

| Section | Description |
|---------|-------------|
| **Process README** | Fetches and indexes a repository README into the vector store (Pinecone) |
| **Query Repositories** | Write a query about a repository and get answers via Orchestrator (RAG + GitHub Agent) |

---

### 📝 Process README

**Inputs:**
- Repository owner (text input)
- Repository name (text input)

**Actions:**
- Click *"Process README"* to fetch the README from GitHub
- The app will split the README into chunks, calculate embeddings, and load them into Pinecone

**UI Feedback:**
- Success/warning/error messages via Streamlit
- Spinners showing progress in chunking, embedding, and Pinecone upload

### 🔍 Query Repositories

**Inputs:**
- Free-form text area to write a question about a repository

**Actions:**
- Click *"Send query"* to run your question through the Orchestrator agent
- The Orchestrator can run the RAG tool (vector search) and the GitHub agent (MCP) to fetch and combine answers

**Output:**
- The final answer (and any error messages) is displayed under *"Answer"*

### ⚙️ Additional Behavior

- The app attempts to connect to the GitHub MCP server on startup; connection errors appear as warnings in the UI
- Uses an `AsyncRunner` to execute asynchronous tasks (MCP connection, agent building, embeddings) without blocking the Streamlit interface
- For debugging, the app logs messages via the Streamlit logger (appears as messages in the UI)
- The default LLM and allowed agent tools are configured in `app.py` — change them if needed for experiments

## 🤖 Agents and Tools

This section describes the three main agent components in the system (**RAGAgent**, **GitHubAgent**, and **Orchestrator**), how they work, what inputs/outputs they expect, and guidance on when to use each one.

### 🧠 RAGAgent

**Purpose:**
- Provide high-level, contextual answers by searching the vector database (Pinecone) for relevant README chunks and synthesizing results with the LLM

**When to use it:**
- Ideal for questions about project purpose, architecture, configuration, usage examples, or any topic where README-derived context is sufficient

**Workflow:**
- Accepts a free-text query
- Embeds the query using the project's embedder (SentenceTransformer)
- Performs a nearest-neighbor search in the vector store returning the top k chunks
- Assembles retrieved chunks into a relevance-aware context payload and calls the LLM to produce a concise, context-grounded answer

**Inputs / Outputs:**
- **Input**: plain query string
- **Output**: synthesized answer string (optionally with citations or snippets from retrieved chunks)

**Limitations:**
- ❌ Not suitable for answering questions that require real-time or file-level repository data (file contents, specific lines, git history) — use **GitHubAgent** for those

### 🔗 GitHubAgent (GitHubMCPAgent + MCPTool)

**Purpose:**
- Provide granular, up-to-date repository information by interacting with a local GitHub MCP server exposed via stdio

**What it provides:**
- Programmatic access to a set of server-exposed tools:
  - 📂 File listing
  - 📄 File content reading
  - 🔎 Repository search
  - 📊 Diffs
  
  Tools are discovered dynamically via MCP session

**How it works:**
- Connects to a local MCP server (container started with Docker) using a PAT available to the server
- Calls `list_tools()` to enumerate available MCP capabilities and wraps each tool as an MCPTool usable by agents
- When invoked, an MCPTool formats JSON-like action input, calls `session.call_tool(tool_name, args)` and parses the tool output into an observation usable for the agent

**Inputs / Outputs:**
- **Input**: structured JSON-like action input (tool-specific fields) or human-readable prompts delegated by an orchestrating agent
- **Output**: raw tool observation (string/JSON), post-processed by `process_tool_output` utilities to keep results consistent for the LLM

**Security and operational notes:**
- 🔐 Requires a valid `GITHUB_TOKEN` provided to the MCP container
- ⏱️ Calls are remote (container ↔ host) and can introduce latency
- 💡 Prefer RAG for cheap local lookups and GitHubAgent for authoritative file-level queries
- ⚠️ Tools can expose sensitive operations; validate inputs and restrict allowed tools where appropriate

### 🎼 Orchestrator

**Purpose:**
- Compose tools and LLMs into a single **ReAct**-style agent that decides when to call RAG or GitHub tools and how to combine results into a final answer

**Main responsibilities:**
- Build an `AgentExecutor` that registers available tools (RAGAgent, MCPTools, GitHubExecTool)
- Implement a structured prompt template that drives the loop: **Thought** → **Action** → **Action Input** → **Observation**
- Enforce iteration limits and timeouts to avoid runaway tool invocation
- Route results from tools back to the LLM and produce the final response shown to the user

**How it chooses tools:**
- The prompt instructs the LLM to:
  - ✅ Prefer **RAG** when README/context is sufficient
  - ✅ Call **GitHub** tools when the question requires file contents, specific lines, or live repository state

**Asynchronous integration and runtime:**
- The Orchestrator can combine synchronous LLM calls and asynchronous MCP calls
- The app uses an `AsyncRunner` with background event loop to run asynchronous operations without blocking Streamlit
- Agent execution captures and surfaces errors from tools; the Orchestrator handles retries, timeouts, and fallbacks where configured

**Example flow:**

```
1️⃣ User query arrives at the Orchestrator
   ↓
2️⃣ Orchestrator invokes RAGAgent to retrieve relevant README context
   ↓
3️⃣ LLM analyzes context and decides a file-level check is needed
   → Calls GitHub MCP tool (via GitHubAgent)
   ↓
4️⃣ GitHub tool returns file content
   ↓
5️⃣ Orchestrator sends combined context to LLM for final answer
```

**Guidance:**
- ⚡ Adjust allowed tools and iteration count depending on typical queries (fewer iterations = faster, safer)
- 💰 Use RAGAgent for **low-cost** contextual responses
- ✔️ Use GitHubAgent when **correctness** and **up-to-date details** matter
