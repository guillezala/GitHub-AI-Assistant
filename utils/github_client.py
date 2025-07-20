from github import Github, GithubException
import os, base64

class GitHubClient:
    def __init__(self, token: str = None):
        token = token or os.getenv("GITHUB_TOKEN")
        self.client = Github(token)

    def fetch_readme(self, owner: str, repo: str) -> str:
        try:
            repository = self.client.get_repo(f"{owner}/{repo}")
            content = repository.get_readme()   
            return base64.b64decode(content.content).decode("utf-8")
        except GithubException as e:
            print(f"[ERROR] GitHub API error: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
        return ""
    
    def get_repo_metadata(self, owner: str, repo: str) -> dict:
        repository = self.client.get_repo(f"{owner}/{repo}")
        return {
            "name": repository.name,
            "full_name": repository.full_name,
            "url": repository.html_url,
            "description": repository.description,
            "created_at": repository.created_at.isoformat(),
            "updated_at": repository.updated_at.isoformat(),
            "language": repository.language
        }