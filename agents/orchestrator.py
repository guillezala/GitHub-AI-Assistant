from typing import List, Tuple, Callable, Any, ClassVar
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_community.chat_models import ChatOllama
import asyncio, json, re

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

def build_agent(tools, prompt, max_iterations=3) -> AgentExecutor:
    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors = True,
        max_iterations=max_iterations,
        return_intermediate_steps=True,
        early_stopping_method="generate" 

    )

class Orchestrator():
    def __init__(
        self,
        tools: List[BaseTool],
        llm: Any,
        logger: Callable[[str], None] = print,
        timeout_s: float = 300.0
    ):
        self.agents = tools
        self.llm = llm
        self.logger = logger
        self.timeout_s = timeout_s

    @staticmethod
    def _strip_json(txt: str) -> dict:
        txt = re.sub(r"^```json|^```|```$", "", txt.strip(), flags=re.MULTILINE)
        try: return json.loads(txt)
        except: return {}

    def evaluate(self, answer: str, query: str) -> bool:
        prompt = (
                "Tu tarea es evaluar si la respuesta resuelve la consulta del usuario.\n"
                "Responde ÚNICAMENTE con uno de estos JSON, sin ningún texto adicional:\n"
                '{"ok": true}\n'
                'o\n'
                '{"ok": false}\n'
                "No expliques nada más.\n\n"
                f"PREGUNTA:\n{query}\n\nRESPUESTA:\n{answer}\n\n"
                "¿La respuesta resuelve la consulta? Devuelve solo el JSON."
            )
        try:
            out = self.llm.invoke(prompt)
            data = self._strip_json(getattr(out, "content", str(out)))
            return bool(data.get("ok", False))
        except Exception:
            return False

    async def _call_agent(self, agent: BaseTool, query: str) -> str:
        try:
            has_async_impl = callable(getattr(agent, "_arun", None))
            if has_async_impl:
                coro = agent.arun(query)      # usa _arun si existe
            else:
                coro = asyncio.to_thread(agent.run, query)  # ejecuta _run en hilo

            task = asyncio.create_task(coro)
            return await asyncio.wait_for(asyncio.shield(task), timeout=self.timeout_s)

        except asyncio.TimeoutError:
            print(f"[{getattr(agent,'name','agent')}] timeout tras {self.timeout_s:.1f}s")
            raise Exception(f"Timeout tras {self.timeout_s:.1f}s")
        except Exception as e:
            self.logger(f"[{getattr(agent,'name','agent')}] error: {e}")
            print(e)
            raise e

    async def _arun(self, query: str) -> str:
        for name, agent in self.agents:
            self.logger(f"[Orquestador] Ejecutando {name}…")
            try:
                resp = await self._call_agent(agent, query)
            except Exception as e:
                print(f"Exception {e}")
                continue
            if self.evaluate(resp, query):
                self.logger(f"[Orquestador] {name} aceptado.")
                return resp
            self.logger(f"[Orquestador] {name} rechazado.")
        self.logger("[Orquestador] Ningún agente válido.")
        return "Lo siento, no he podido encontrar una respuesta adecuada."

    def _run(self, query: str) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._arun(query))
        else:
            return loop.run_until_complete(self._arun(query))