import json
import re
from typing import Dict, Optional

class QueryAnalyzer:
    def __init__(self, llm, logger=None):
        self.llm = llm
        self.logger = logger
        
        # Palabras clave para detección rápida
        self.keywords_codigo = {
            'programacion': ['código', 'code', 'programar', 'programming', 'script', 'función', 'function', 
                           'variable', 'clase', 'class', 'método', 'method', 'algoritmo', 'algorithm'],
            'github': ['github', 'git', 'repositorio', 'repository', 'repo', 'pull request', 'pr', 
                      'commit', 'branch', 'fork', 'clone', 'merge'],
            'open_source': ['open source', 'código abierto', 'opensource', 'libre', 'free software',
                          'licencia', 'license', 'contribuir', 'contribute'],
            'lenguajes': ['python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                        'typescript', 'html', 'css', 'sql', 'bash', 'shell']
        }

    def _quick_keyword_check(self, query: str) -> Dict[str, float]:
        """Análisis rápido basado en palabras clave"""
        query_lower = query.lower()
        scores = {
            'programacion': 0.0,
            'github': 0.0,
            'codigo_abierto': 0.0
        }
        
        # Buscar palabras clave
        for category, keywords in self.keywords_codigo.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if category == 'programacion':
                scores['programacion'] = min(matches * 0.3, 1.0)
            elif category == 'github':
                scores['github'] = min(matches * 0.4, 1.0)
            elif category in ['open_source', 'lenguajes']:
                scores['codigo_abierto'] += min(matches * 0.2, 0.5)
        
        scores['codigo_abierto'] = min(scores['codigo_abierto'], 1.0)
        return scores

    def _extract_repository(self, query: str) -> Optional[str]:
        """Extrae nombre de repositorio de la consulta"""
        # Patrones comunes para repositorios
        patterns = [
            r'github\.com/([^/\s]+/[^/\s]+)',  # github.com/user/repo
            r'(?:repo|repositorio)[:\s]+([^\s]+/[^\s]+)',  # repo: user/repo
            r'([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)(?:\s|$)',  # user/repo format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                repo = match.group(1)
                if '/' in repo and len(repo.split('/')) == 2:
                    return repo
        return None

    def analyze_query(self, query: str) -> Dict:
        """
        Analiza si la pregunta es sobre código abierto, GitHub o programación.
        Combina análisis por keywords y LLM para mayor precisión.
        """
        # Análisis rápido por keywords
        keyword_scores = self._quick_keyword_check(query)
        repository = self._extract_repository(query)
        
        # Prompt mejorado y más específico
        prompt = f"""Eres un clasificador con sesgo POSITIVO hacia la temática de programación.
Asume por defecto que la pregunta sí está relacionada con programación/tecnología.
Solo marca todo en false cuando sea CLARAMENTE ajena (cocina, deportes, viajes, etc.).

Categorías:
1. Código abierto / open source (proyectos, librerías o frameworks públicos, licencias, contribuciones públicas)
2. GitHub (plataforma, repositorios, Issues/PRs, Actions, Gists, Releases)
3. Programación (desarrollo de software, lenguajes, código, APIs, debugging, compilación, arquitectura, herramientas de dev)

REGLAS:
- Responde ÚNICAMENTE con un JSON válido y cerrado. Nada fuera del JSON.

Pregunta: "{query}"

Formato de respuesta esperado:
{{
  "es_codigo_abierto": boolean,
  "es_github": boolean,
  "es_programacion": boolean,
  "repositorio": "usuario/repo" o null,
  "confianza": número entre 0 y 1,
  "razonamiento": "explicación breve"
}}"""

        try:
            # Generar respuesta del LLM
            response = self.llm.invoke(prompt)
            
            # Limpiar respuesta (quitar markdown, espacios, etc.)
            response = self._clean_llm_response(response.content)
            
            # Parsear JSON
            llm_analysis = json.loads(response)
            
            # Validar estructura
            llm_analysis = self._validate_analysis(llm_analysis)
            
            # Combinar con análisis de keywords para mayor robustez
            final_analysis = self._combine_analyses(llm_analysis, keyword_scores, repository)
            
        except Exception as e:
            if self.logger:
                self.logger(f"[Orquestador] Error en análisis LLM: {e}")
            
            # Fallback: usar solo análisis de keywords
            final_analysis = {
                "es_codigo_abierto": keyword_scores['codigo_abierto'] > 0.3,
                "es_github": keyword_scores['github'] > 0.3,
                "es_programacion": keyword_scores['programacion'] > 0.3,
                "repositorio": repository,
                "confianza": max(keyword_scores.values()) if any(v > 0.3 for v in keyword_scores.values()) else 0.1,
                "razonamiento": "Análisis basado en keywords (LLM falló)"
            }
        
        """if self.logger:
            self.logger(f"[Orquestador] Análisis final: {final_analysis}")"""
        
        return final_analysis

    def _clean_llm_response(self, response: str) -> str:
        """Limpia la respuesta del LLM para extraer solo el JSON"""
        # Quitar bloques de código markdown
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)

        # Eliminar líneas que no empiezan con { o terminan con }
        lines = response.splitlines()
        json_lines = [line for line in lines if '{' in line or '}' in line or ':' in line]
        response = "\n".join(json_lines)
        
        # Buscar el JSON en la respuesta
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        if response.count('{') > response.count('}'):
            response += '}'
        
        return response

    def _validate_analysis(self, analysis: Dict) -> Dict:
        """Valida y corrige la estructura del análisis"""
        required_fields = {
            "es_codigo_abierto": False,
            "es_github": False,
            "es_programacion": False,
            "repositorio": None,
            "confianza": 0.0,
            "razonamiento": ""
        }
        
        # Asegurar que todos los campos existen
        for field, default in required_fields.items():
            if field not in analysis:
                analysis[field] = default
        
        # Validar tipos
        analysis["es_codigo_abierto"] = bool(analysis["es_codigo_abierto"])
        analysis["es_github"] = bool(analysis["es_github"])
        analysis["es_programacion"] = bool(analysis["es_programacion"])
        analysis["confianza"] = max(0.0, min(1.0, float(analysis["confianza"])))
        
        return analysis

    def _combine_analyses(self, llm_analysis: Dict, keyword_scores: Dict, repository: Optional[str]) -> Dict:
        """Combina análisis del LLM con análisis de keywords"""
        # Si el LLM tiene baja confianza, usar keywords como respaldo
        if llm_analysis["confianza"] < 0.4:
            llm_analysis["es_codigo_abierto"] = llm_analysis["es_codigo_abierto"] or keyword_scores['codigo_abierto'] > 0.3
            llm_analysis["es_github"] = llm_analysis["es_github"] or keyword_scores['github'] > 0.3
            llm_analysis["es_programacion"] = llm_analysis["es_programacion"] or keyword_scores['programacion'] > 0.3
            
            # Ajustar confianza basada en keywords
            max_keyword_score = max(keyword_scores.values())
            if max_keyword_score > 0.3:
                llm_analysis["confianza"] = max(llm_analysis["confianza"], max_keyword_score)
        
        # Usar repositorio extraído si el LLM no lo detectó
        if not llm_analysis["repositorio"] and repository:
            llm_analysis["repositorio"] = repository
        
        return llm_analysis

    def is_relevant_query(self, analysis, min_confidence: float = 0.4) -> bool:
        """Determina si la consulta es relevante para el sistema"""
        is_relevant = (
            analysis["es_codigo_abierto"] or 
            analysis["es_github"] or 
            analysis["es_programacion"]
        ) and analysis["confianza"] >= min_confidence
        
        return is_relevant
