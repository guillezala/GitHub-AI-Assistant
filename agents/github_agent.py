import asyncio
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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

        # 1) Asegura token
        if not self.pat:
            raise ValueError("GITHUB token no configurado en entorno.")

        # 2) Si no te pasan args, construye los de docker run
        if args is None:
            args = [
                "run", "--rm", "-i",
                "-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={self.pat}",
                "ghcr.io/github/github-mcp-server",
                "--enable-command-logging", "--log-file", "/tmp/mcp.log",
                "stdio",
            ]

        # 3) El env aquí aplica al proceso 'docker' del host (no al contenedor),
        #    así que no confíes en él para el token; ya lo pasamos con -e arriba.
        env = dict(os.environ)
        if extra_env:
            env.update(extra_env)

        params = StdioServerParameters(command=self.server_cmd, args=args, env=env)

        # 4) Abre el proceso y el session MCP
        try:
            stdio = await self.exit_stack.enter_async_context(stdio_client(params))
            read, write = stdio
            self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
            return await self.list_tools()
        except Exception:
            # TIP: activa logs del servidor para ver por qué se cierra
            # Añade antes de "stdio" en args:
            #   "--enable-command-logging", "--log-file", "/tmp/mcp.log"
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

    async def close(self):
        try:
            await self.exit_stack.aclose()
        finally:
            self.session = None
            self.exit_stack = AsyncExitStack()
