from langchain.tools import BaseTool
from langchain.agents import AgentExecutor


class GitHubExecTool(BaseTool):
    name: str = "GitHubAgent"
    description: str = "Usa el agente GitHub (MCP) para responder consultas de repos."
    executor: AgentExecutor

    def _run(self, query: str) -> str:
        res = self.executor.invoke({"input": query})
        return res["output"]

    async def _arun(self, query: str) -> str:
        res = await self.executor.ainvoke({"input": query}, include_run_info=True, return_intermediate_steps=False)
        return res["output"]