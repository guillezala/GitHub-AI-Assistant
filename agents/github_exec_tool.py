from langchain.tools import BaseTool
from langchain.agents import AgentExecutor


class GitHubExecTool(BaseTool):
    name: str = "GitHubAgent"
    description: str = "Usa el agente GitHub para responder consultas sobre repositorios. Puede acceder al contenido y detalles de git de los repositorios"
    executor: AgentExecutor

    def _run(self, query: str) -> str:
        res = self.executor.invoke({"input": query})
        return res["output"] if isinstance(res, dict) else res

    async def _arun(self, query: str) -> str:
        res = await self.executor.ainvoke({"input": query}, include_run_info=True, return_intermediate_steps=False)
        return res["output"] if isinstance(res, dict) else res