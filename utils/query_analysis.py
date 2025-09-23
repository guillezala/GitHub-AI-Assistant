import json
import re
from typing import Dict, Optional

class QueryAnalyzer:
    def __init__(self, llm, logger=None):
        self.llm = llm
        self.logger = logger
        
        # Palabras clave para detección rápida
        self.keywords_codigo = {
            'programming': ['código', 'code', 'programar', 'programming', 'script', 'función', 'function', 
                           'variable', 'clase', 'class', 'método', 'method', 'algoritmo', 'algorithm'],
            'github': ['github', 'git', 'repositorio', 'repository', 'repo', 'pull request', 'pr', 
                      'commit', 'branch', 'fork', 'clone', 'merge'],
            'open_source': ['open source', 'código abierto', 'opensource', 'libre', 'free software',
                          'licencia', 'license', 'contribuir', 'contribute'],
            'lenguages': ['python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                        'typescript', 'html', 'css', 'sql', 'bash', 'shell']
        }

    def _quick_keyword_check(self, query: str) -> Dict[str, float]:
        """Análisis rápido basado en palabras clave"""
        query_lower = query.lower()
        scores = {
            'programming': 0.0,
            'github': 0.0,
            'open_source': 0.0
        }
        
        for category, keywords in self.keywords_codigo.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if category == 'programming':
                scores['programming'] = min(matches * 0.3, 1.0)
            elif category == 'github':
                scores['github'] = min(matches * 0.4, 1.0)
            elif category in ['open_source', 'lenguages']:
                scores['open_source'] += min(matches * 0.2, 0.5)
        
        scores['open_source'] = min(scores['open_source'], 1.0)
        return scores

    def _extract_repository(self, query: str) -> Optional[str]:
        patterns = [
            r'github\.com/([^/\s]+/[^/\s]+)', 
            r'(?:repo|repositorio)[:\s]+([^\s]+/[^\s]+)',  
            r'([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)(?:\s|$)',  
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                repo = match.group(1)
                if '/' in repo and len(repo.split('/')) == 2:
                    return repo
        return None

    def analyze_query(self, query: str) -> Dict:
        keyword_scores = self._quick_keyword_check(query)
        repository = self._extract_repository(query)
        
        # Prompt mejorado y más específico
        prompt = f"""You are a classifier with a POSITIVE bias towards programming topics.
Assume by default that the question IS related to programming/technology.
Only mark everything as false when it's CLEARLY unrelated (cooking, sports, travel, etc.).
Categories:

Open source (projects, public libraries or frameworks, licenses, public contributions)
GitHub (platform, repositories, Issues/PRs, Actions, Gists, Releases)
Programming (software development, languages, code, APIs, debugging, compilation, architecture, dev tools)

RULES:

Respond ONLY with a valid and closed JSON. Nothing outside the JSON.

Question: "{query}"
Expected response format:
{{
"open_source": boolean,
"github": boolean,
"programming": boolean,
"repository": "user/repo" or null,
"trust": number between 0 and 1,
"reasoning": "brief explanation"
}}"""

        try:
            response = self.llm.invoke(prompt)
            
            response = self._clean_llm_response(response.content)
            
            llm_analysis = json.loads(response)
            
            llm_analysis = self._validate_analysis(llm_analysis)
            
            final_analysis = self._combine_analyses(llm_analysis, keyword_scores, repository)
            
        except Exception as e:
            if self.logger:
                self.logger(f"[Orchestrator] Error in LLM analysis: {e}")
            
            final_analysis = {
                "open_source": keyword_scores['open_source'] > 0.3,
                "github": keyword_scores['github'] > 0.3,
                "programming": keyword_scores['programming'] > 0.3,
                "repository": repository,
                "trust": max(keyword_scores.values()) if any(v > 0.3 for v in keyword_scores.values()) else 0.1,
                "reasoning": "Analysis based on keywords (LLM failed)"
            }
        
        return final_analysis

    def _clean_llm_response(self, response: str) -> str:

        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)

        lines = response.splitlines()
        json_lines = [line for line in lines if '{' in line or '}' in line or ':' in line]
        response = "\n".join(json_lines)
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        if response.count('{') > response.count('}'):
            response += '}'
        
        return response

    def _validate_analysis(self, analysis: Dict) -> Dict:
        """Valida y corrige la estructura del análisis"""
        required_fields = {
            "open_source": False,
            "github": False,
            "programming": False,
            "repository": None,
            "trust": 0.0,
            "reasoning": ""
        }
        
        for field, default in required_fields.items():
            if field not in analysis:
                analysis[field] = default
        
        analysis["open_source"] = bool(analysis["open_source"])
        analysis["github"] = bool(analysis["github"])
        analysis["programming"] = bool(analysis["programming"])
        analysis["trust"] = max(0.0, min(1.0, float(analysis["trust"])))
        
        return analysis

    def _combine_analyses(self, llm_analysis: Dict, keyword_scores: Dict, repository: Optional[str]) -> Dict:

        if llm_analysis["confianza"] < 0.4:
            llm_analysis["open_source"] = llm_analysis["open_source"] or keyword_scores['open_source'] > 0.3
            llm_analysis["github"] = llm_analysis["github"] or keyword_scores['github'] > 0.3
            llm_analysis["programming"] = llm_analysis["programming"] or keyword_scores['programming'] > 0.3
            

            max_keyword_score = max(keyword_scores.values())
            if max_keyword_score > 0.3:
                llm_analysis["trust"] = max(llm_analysis["trust"], max_keyword_score)
        

        if not llm_analysis["repository"] and repository:
            llm_analysis["repository"] = repository
        
        return llm_analysis

    def is_relevant_query(self, analysis, min_confidence: float = 0.4) -> bool:
        is_relevant = (
            analysis["open_source"] or 
            analysis["github"] or 
            analysis["programming"]
        ) and analysis["trust"] >= min_confidence
        
        return is_relevant
