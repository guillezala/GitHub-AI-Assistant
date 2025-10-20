from typing import List, Callable, Any
from langchain.tools import BaseTool
import asyncio

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate



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

    async def build_orchestrator(self) -> AgentExecutor:
        llm = self.llm
        tools=[]
        for _, agent in self.agents:
            tools.append(agent)

        REACT_PROMPT = """ Answer the following questions as best you can. You have access to the following tools
            {tools}

            Use the following format:

            Question: the input question
            Thought: reasoning
            Action: one of [{tool_names}]
            Action Input: A brief reasoning based on the Question and with all the details needed\n
            Observation: result of the action
            ... (repeat as needed)

            Thought: I now know the final answer
            Final Answer: the final answer

            Rules:
            - End with 'Final Answer:' once you can fully answer.
            - Keep observations concise and relevant.

            Question: {input}
            Thought:{agent_scratchpad}"""
        

        prompt = PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
            template=REACT_PROMPT,
        )


        agent = create_react_agent(llm, tools, prompt)
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