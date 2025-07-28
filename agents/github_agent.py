from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from github import Github
from langchain.tools import BaseTool
from pydantic import Field
import os
class GitHubTool(Tool):
    """
    Herramienta LangChain para interactuar con la API de GitHub usando PyGithub.
    """
    def __init__(self, gh_token: str = None):
        super().__init__(
            name="github_tool",
            func=self.run,
            description=(
                "Permite buscar repositorios, issues y descargar archivos de GitHub. "
                "Uso: especificar la acción como 'search_repos', 'get_file', 'list_issues'. "
                "Parámetros: action:str, repo:str, query_or_path:str"
            )
        )
        gh_token = gh_token or os.getenv("GITHUB_TOKEN")
        object.__setattr__(self, "client", Github(gh_token))

    def run(self, **kwargs):
        action = kwargs.get("action")
        repo_name = kwargs.get("repo")
        param = kwargs.get("query_or_path")

        if action == "search_repos":
            return self.search_repos(param)
        elif action == "list_issues":
            return self.list_issues(repo_name)
        elif action == "get_file":
            return self.get_file(repo_name, param)
        else:
            return f"Acción desconocida: {action}"

    def search_repos(self, query: str):
        results = self.client.search_repositories(query)
        top = results[:5]
        return [r.full_name for r in top]

    def list_issues(self, repo_full_name: str):
        repo = self.client.get_repo(repo_full_name)
        issues = repo.get_issues(state="open")
        return [issue.title for issue in issues[:5]]

    def get_file(self, repo_full_name: str, path: str):
        repo = self.client.get_repo(repo_full_name)
        file = repo.get_contents(path)
        return file.decoded_content.decode()


class GitHubAgent(BaseTool):
    name: str = Field(default="GitHubAgent")
    description: str = Field(default="Interactúa con la API de GitHub y responde preguntas sobre repositorios, issues y archivos.")
    github_tool: object = Field(...)
    llm: object = Field(...)
    prompt_template: object = Field(default=None)

    def _run(self, query: str) -> str:
        # Usa el prompt personalizado si existe, si no uno por defecto
        prompt = self.prompt_template or PromptTemplate(
            input_variables=["input"],
            template=(
                "Eres un agente que interactúa con GitHub. "
                "Tienes acceso a la herramienta github_tool. \n"
                "Pregunta: {input}"
            )
        )
        # Inicializa el agente LangChain
        from langchain.agents import initialize_agent
        agent_executor = initialize_agent(
            [self.github_tool],
            self.llm,
            agent="zero-shot-react-description",
            verbose=False,
            prompt=prompt,
        )
        # Ejecuta la consulta
        return agent_executor.run(query)