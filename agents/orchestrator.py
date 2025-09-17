from typing import List, Tuple, Callable, Any, ClassVar
from langchain.tools import BaseTool
import asyncio, json, re

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate

class Orchestrator():
    def __init__(
        self,
        tools: List[tuple[str, BaseTool]],
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

    async def build_orchestrator(self) -> AgentExecutor:
        llm = self.llm
        tools=[]
        for _, agent in self.agents:
            tools.append(agent)

        REACT_PROMPT = ChatPromptTemplate.from_messages([
            ("system",
            "You are a helpful orchestrator. Use tools only when needed. "
            "Cite tool names exactly as provided."),
            ("human",
            "Answer the question using the available tools:\n\n{tools}\n\n"
            "Format:\n"
            "Question: the input question\n"
            "Thought: reasoning\n"
            "Action: one of [{tool_names}]\n"
            "Action Input: input for the action\n"
            "Observation: result of the action\n"
            "... (repeat as needed)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer\n\n"
            "Rules:\n"
            "- End with 'Final Answer:' once you can fully answer.\n"
            "- Do not call the same tool more than 2 times in a row.\n"
            "- Keep observations concise and relevant.\n\n"
            "Question: {input}\n"
            "Thought:{agent_scratchpad}")
        ])


        agent = create_react_agent(llm, tools, REACT_PROMPT)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors = "Fix the format. Stick to the given prompt. Make sure there is a Thought, Action and Action Input.",
            max_iterations=6,
            return_intermediate_steps=True,
            early_stopping_method="generate" 

        )
        return executor