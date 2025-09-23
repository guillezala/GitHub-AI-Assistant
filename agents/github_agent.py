import asyncio
import json
from contextlib import AsyncExitStack
from typing import Optional, Any, Dict, Iterable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_ollama import ChatOllama
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

from utils.process_tool_output import process_tool_output

import os

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

        listed = await self.session.list_tools()
        available = {t.name: t for t in listed.tools}

        if allowed_tools is None:
            selected = available.values()
        else:
            selected = [available[name] for name in allowed_tools if name in available]

        tools = []
        for t in selected:
            
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

        llm = ChatOllama(model=model, temperature=temperature)

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
                - Do NOT add an optional parameter to the tool if not necessary or explicitly mentioned in the input.
                - Every tool input must have "repo" and "owner" parameters.

                Begin!

                Question: {input}
                Thought: {agent_scratchpad}
                """

        prompt = PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
            template=REACT_PROMPT,
        )

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
    _cache: Dict[str, str] = {}

    def _run(self, tool_input, run_manager=None) -> str:

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

        
        result = await self.session.call_tool(self.mcp_tool_name, args)

        processed_output = process_tool_output(self.mcp_tool_name, result)

        return processed_output
