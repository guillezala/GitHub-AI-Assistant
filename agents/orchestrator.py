from langchain.tools import BaseTool
from pydantic import Field
from typing import List, Tuple, Callable
from logging import Logger
import logging

import asyncio

class OrchestratorAgent(BaseTool):
    name: str = "OrchestratorAgent"
    description: str = "Orquesta múltiples agentes (RAG, GitHub, etc.) y decide cuál usar según la consulta y la calidad de la respuesta."
    agents: List[Tuple[str, BaseTool]] = Field(...)
    llm: object = Field(...)
    logger: object = Field(...)

    def evaluate(self, answer: str, query: str) -> bool:
        """
        Devuelve True si el LLM considera la respuesta satisfactoria.
        """
        prompt = (
            f"La pregunta: '''{query}'''.\n"
            f"La respuesta: '''{answer}'''.\n"
            "¿Es satisfactoria? Devuélveme un JSON: {\"ok\": true} o {\"ok\": false}."
        )
        resp = self.llm.generate([prompt]).generations[0][0].text.strip()
        ok = resp.startswith("{\"ok\": true}")
        return ok
    
    
    async def _arun(self, query: str) -> str:
        for name, agent in self.agents:
            self.logger(f"[Orquestador] Ejecutando {name} (async)…")
            try:
                resp = await agent.arun(query)
            except Exception as e:
                self.logger(f"[{name}] error: {e}")
                continue

            # <-- aquí quitas el await
            if self.evaluate(resp, query):
                self.logger(f"[Orquestador] {name} aceptado.")
                return resp
            else:
                self.logger(f"[Orquestador] {name} rechazado.")

        self.logger("[Orquestador] Ningún agente válido.")
        return "Lo siento, no he podido encontrar una respuesta adecuada."

    """def _run(self, query: str) -> str:
        for name, agent in self.agents:
            self.logger(f"[Orquestador] Ejecutando {name}...")
            try:
                resp = agent.run(query)
            except Exception as e:
                self.logger(f"[{name}] error: {e}")
                continue
            if self.evaluate(resp, query):
                self.logger(f"[Orquestador] {name} aceptado.")
                return resp
            else:
                self.logger(f"[Orquestador] {name} rechazado.")
        self.logger("[Orquestador] Ningún agente válido.")
        return "Lo siento, no he podido encontrar una respuesta adecuada." """
    
    def _run(self, query: str) -> str:
        return asyncio.run(self._arun(query))