import asyncio
import json
from contextlib import AsyncExitStack
from typing import Optional, Any, Dict, Iterable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolRequest

from langchain_community.chat_models import ChatOllama
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

import os


MAX_ITEMS = 20
MAX_CHARS = 1800 

def summarize_list_of_dicts(items: list[dict]) -> str:
    total = len(items)
    # Heuristics for common GitHub entities
    lines = []
    for it in items[:MAX_ITEMS]:
        if isinstance(it, dict):
            num = it.get("number") or it.get("id") or "-"
            state = it.get("state") or it.get("status") or ""
            title = it.get("title") or it.get("name") or it.get("path") or "(sin título)"
            line = f"- #{num} {state} — {title}" if state else f"- #{num} — {title}"
            lines.append(line)
        else:
            lines.append(f"- {str(it)[:120]}")
    more = ""
    if total > MAX_ITEMS:
        more = f"\n… y {total - MAX_ITEMS} más."
    header = f"Total: {total}\n"
    return header + "\n".join(lines) + more

def summarize_json(obj: Any) -> str:
    try:
        if isinstance(obj, list):
            if obj and isinstance(obj[0], dict):
                return summarize_list_of_dicts(obj)
            # fallback: truncate generic list
            s = "\n".join([json.dumps(x, ensure_ascii=False) for x in obj[:MAX_ITEMS]])
            extra = f"\n… y {len(obj) - MAX_ITEMS} más." if len(obj) > MAX_ITEMS else ""
            return f"Total: {len(obj)}\n" + s + extra
        if isinstance(obj, dict):
            # common pattern: { items: [...] }
            if isinstance(obj.get("items"), list):
                return summarize_list_of_dicts(obj["items"])
                # fallback to pretty JSON
        return json.dumps(obj, ensure_ascii=False)[:MAX_CHARS]
    except Exception:
        return json.dumps(obj, ensure_ascii=False)[:MAX_CHARS]
class GitHubMCPAgent:
    def __init__(self, server_cmd: str = "docker", pat_env: str = "GITHUB_TOKEN"):
        self.server_cmd = server_cmd
        self.pat = os.environ.get(pat_env)
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect(self, extra_env: dict | None = None, args: list[str] | None = None):
        if self.session is not None:
            return  # ya conectados

        if not self.pat:
            raise ValueError("GITHUB token no configurado en entorno.")

        if args is None:
            args = [
                "run", "--rm", "-i",
                "-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={self.pat}",
                "ghcr.io/github/github-mcp-server",
                "--enable-command-logging", "--log-file", "/tmp/mcp.log",
                "stdio",
            ]

        env = dict(os.environ)
        if extra_env:
            env.update(extra_env)

        params = StdioServerParameters(command=self.server_cmd, args=args, env=env)

        try:
            stdio = await self.exit_stack.enter_async_context(stdio_client(params))
            read, write = stdio
            self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
            return await self.list_tools()
        except Exception:
            raise

    async def ensure_connected(self):
        if self.session is None:
            await self.connect()

    async def list_tools(self):
        assert self.session is not None, "No conectado"
        tools = (await self.session.list_tools()).tools
        return [{"name": t.name, "description": t.description, "inputSchema": t.inputSchema} for t in tools]

    async def call(self, tool_name: str, tool_args: dict):
        await self.ensure_connected()
        assert self.session is not None, "No conectado"
        return await self.session.call_tool(tool_name, tool_args)
    
    async def build_executor(
        self,
        allowed_tools: Optional[Iterable[str]] = None,
        model: str = "llama3.1",
        temperature: float = 0.0,
        max_iterations: int = 8,
        prompt_template: Optional[str] = None,
    ) -> AgentExecutor:
        """Crea un AgentExecutor ReAct (LangChain) que usa las tools del servidor MCP."""
        await self.ensure_connected()
        assert self.session is not None

        # 1) Descubre tools del servidor MCP
        listed = await self.session.list_tools()
        available = {t.name: t for t in listed.tools}

        # 2) Filtra por whitelist (si se pasa). Si None, usa todas (no recomendado en prod)
        if allowed_tools is None:
            selected = available.values()
        else:
            selected = [available[name] for name in allowed_tools if name in available]

        # 3) Crea wrappers LangChain -> MCP
        tools = []
        for t in selected:
            # Intenta renderizar un resumen del JSON Schema para guiar al modelo
            schema_hint = ""
            try:
                schema_dict = t.inputSchema or {}
                # Muestra solo propiedades y required si existen
                props = schema_dict.get("properties") or {}
                req = schema_dict.get("required") or []
                if props or req:
                    schema_hint = "\n\nInput JSON schema (summary):\n"
                    if props:
                        schema_hint += "properties: " + ", ".join(f"{k}" for k in props.keys()) + "\n"
                    if req:
                        schema_hint += "required: " + ", ".join(req) + "\n"
            except Exception:
                pass

            # Extra guidance for the LLM about summarized outputs
            extra_desc = (
                "\n\nNote: Outputs may be summarized and limited in length "
                "to the most relevant fields and a small number of items. "
                "Totals are included. Do not repeat the same call if the observation already answers the question."
            )

            tools.append(
                MCPTool(
                    name=t.name,
                    description=(t.description or f"Tool {t.name} from MCP server") + schema_hint + extra_desc,
                    session=self.session,
                    mcp_tool_name=t.name,
                )
            )

        # 4) LLM de Ollama
        llm = ChatOllama(model=model, temperature=temperature)

        # 5) Prompt ReAct sobrio
        REACT_PROMPT = """You are an expert agent that uses tools to answer questions.

        You have access to the following tools:
        {tools}

        On each turn, produce EXACTLY ONE of the following:

        (1) A single tool call:
        Thought: <brief reasoning>
        Action: <one of [{tool_names}]>
        Action Input: <valid JSON, no backticks>
        # IMPORTANT: After printing Action Input, END YOUR MESSAGE IMMEDIATELY.
        # Do NOT add anything after (no more lines, no extra words).

        (2) The final answer:
        Thought: I now know the final answer
        Final Answer: <your answer>

        Rules:
        - Never include both a tool call and a Final Answer in the same turn.
        - Do not wrap JSON in backticks.
        - Use the exact labels: Thought / Action / Action Input / Observation / Final Answer.
        - If the Observation contains a list, summarize it and only mention the most relevant items.
        - If you already have the information requested, respond with Final Answer and stop.
        - Do NOT repeat the same tool call if the observation already contains the answer.

        Begin!

        Question: {input}
        {agent_scratchpad}
        """

        prompt = PromptTemplate.from_template(REACT_PROMPT)

        from langchain import hub
        prompt2 = hub.pull("hwchase17/react")
    
        # 6) Monta el agente ReAct
        agent = create_react_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors="Fix the format. After Action Input, stop immediately. One step only.",
            max_iterations=max_iterations,
        )
        return executor

    async def close(self):
        try:
            await self.exit_stack.aclose()
        finally:
            self.session = None
            self.exit_stack = AsyncExitStack()

class MCPTool(BaseTool):
    name: str
    description: str
    session: ClientSession
    mcp_tool_name: str
    schema: Dict[str, Any] = {}  # opcional: tu JSON Schema para validar

    def _run(self, tool_input, run_manager=None) -> str:
        # Soporta string o dict
        args = tool_input
        if isinstance(args, str):
            stripped = args.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    args = json.loads(stripped)
                except json.JSONDecodeError:
                    args = {"query": args}
            else:
                args = {"query": args}
        return asyncio.run(self._arun(args, run_manager))

    async def _arun(self, tool_input, run_manager=None) -> str:
        args = tool_input
        if isinstance(args, str):
            stripped = args.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    args = json.loads(stripped)
                except json.JSONDecodeError:
                    args = {"query": args}
            else:
                args = {"query": args}

        result = await self.session.call_tool(self.mcp_tool_name, args)

        # Collect outputs by type
        json_objs = []
        text_chunks = []
        other_chunks = []
        for out in getattr(result, "content", []):
            t = getattr(out, "type", "")
            if t == "json":
                try:
                    json_objs.append(getattr(out, "content", {}))
                except Exception:
                    pass
            elif t == "text":
                text_chunks.append(getattr(out, "text", "") or "")
            else:
                other_chunks.append(str(out))

        if json_objs:
            parts = [summarize_json(o) for o in json_objs]
            out = "\n".join(p for p in parts if p).strip()
            # Add a small hint that this is a summarized, sufficient observation
            if out:
                out = out[:MAX_CHARS]
                return out + "\n(Resumen generado; no repitas la misma tool si ya responde la consulta)"

        # No JSON available, return text/other truncated in a safe budget
        combined = "\n".join([*(c for c in text_chunks if c), *(c for c in other_chunks if c)]).strip()
        if combined:
            return combined[:MAX_CHARS]
        return "(sin salida)"
