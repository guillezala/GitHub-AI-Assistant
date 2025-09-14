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

from utils.process_tool_output import process_tool_output

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
    
    async def build_executor(
        self,
        allowed_tools: Optional[Iterable[str]] = None,
        model: str = "llama3.2:3b",
        temperature: float = 0.0,
        max_iterations: int = 8,
        prompt_template: Optional[str] = None,
    ) -> AgentExecutor:
        """Crea un AgentExecutor ReAct (LangChain) que usa las tools del servidor MCP."""
        await self.ensure_connected()
        assert self.session is not None

        # 1) Descubre tools del servidor MCP
        listed = await self.session.list_tools()
        print(listed)
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

            tools.append(
                MCPTool(
                    name=t.name,
                    description=(t.description or f"Tool {t.name} from MCP server") + schema_hint,
                    session=self.session,
                    mcp_tool_name=t.name,
                )
            )

        # 4) LLM de Ollama
        llm = ChatOllama(model=model, temperature=temperature)

        # 5) Prompt ReAct sobrio
        REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:
                {tools}

                Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question

                IMPORTANT:
                - End with "Final Answer:" once you can fully answer the question.
                - Do not call the same tool more than 2 times in a row if it doesn’t help.
                - The tool will return JSON outputs. To create the Observation, summarized it using the relevant information in relation with the input.

                Begin!

                Question: {input}
                Thought: {agent_scratchpad}
                """

        prompt = PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
            template=REACT_PROMPT,
        )

        from langchain import hub
        prompt2 = hub.pull("hwchase17/react")
    
        # 6) Monta el agente ReAct
        agent = create_react_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors = "Fix the format. Stick to the given prompt. Make sure there is a Thought, Action and Action Input.",
            max_iterations=max_iterations,
            return_intermediate_steps=True,
            early_stopping_method="generate" 

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
    _cache: Dict[str, str] = {}

    def _key(self, args: dict) -> str:
        return f"{self.mcp_tool_name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"

    def _run(self, tool_input, run_manager=None) -> str:
        # Soporta string o dict
        args = tool_input
        if isinstance(args, str):
            stripped = args.strip()
            if not (stripped.startswith("{") or stripped.startswith("[")):
                raise ValueError(
                    "Action Input must be valid JSON matching the tool schema. "
                    "Provide an object with the required keys."
                )
            try:
                args = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for Action Input: {e}")

        elif not isinstance(args, dict):
            raise ValueError("Action Input must be a JSON object (dict).")

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

        key = self._key(args)

        if key in self._cache:
            return f"(cached) {self._cache[key]}"
        
        result = await self.session.call_tool(self.mcp_tool_name, args)

        processed_output = process_tool_output(self.mcp_tool_name, result)
        self._cache[key] = processed_output

        hint = "Observation summary (use this to answer; do not call the tool again if sufficient):\n"

        return hint + processed_output
