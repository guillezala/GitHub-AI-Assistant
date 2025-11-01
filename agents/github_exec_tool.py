from langchain.tools import BaseTool
from langchain.agents import AgentExecutor

from pydantic import BaseModel, Field

class ArgsSchema(BaseModel):
    input: str = Field(..., description="Reasoning based on the user query with all the necessary details")

class GitHubExecTool(BaseTool):
    name: str = "GitHubAgent"
    description: str = "Use the GitHub agent to answer queries about repositories. It can access the files and git details of the repositories"
    executor: AgentExecutor
    args_schema: type = ArgsSchema

    def _run(self, input: str) -> str:
        res = self.executor.invoke({"input": input})
        return res["output"] if isinstance(res, dict) else res

    async def _arun(self, input: str) -> str:
        res = await self.executor.ainvoke({"input": input}, include_run_info=True, return_intermediate_steps=False)
        return res["output"] if isinstance(res, dict) else res