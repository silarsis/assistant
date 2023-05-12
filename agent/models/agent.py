from langchain.agents import Tool, LLMSingleActionAgent, AgentExecutor, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.prompts.base import StringPromptValue, PromptValue
from langchain.memory.motorhead_memory import MotorheadMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import AsyncCallbackHandler
from langchain import LLMChain
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.llms import AzureOpenAI, OpenAI
from typing import List, Union, Any
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from transformers import OpenAiAgent, HfAgent
import re
import os

from typing import Callable

TEMPERATURE = 0.2


class CodeAgent:
    def __init__(self):
        self._agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
        
    def run(self, query: str) -> str:
        " Useful for answering questions about code "
        print("Got to the CodeAgent tool")
        return self._agent.run(query, temperature=TEMPERATURE, remote=True)
        
class CodeExplainerAgent:
    def __init__(self):
        self._agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
        
    def run(self, query: str) -> str:
        " Useful for answering questions about code "
        print("Got to the CodeExplainerAgent tool")
        return self._agent.run(f"Please explain the following code:\n{query}", temperature=TEMPERATURE, remote=True)
        
class TransformerToolAgent:
    def __init__(self):
        self._agent = OpenAiAgent(model='text-davinci-003', api_key=os.environ["OPENAI_API_KEY"])
        
    def run(self, query: str) -> str:
        " Useful when you don't know how else to answer "
        print("Got to the general transformer tool")
        return self._agent.run(query, temperature=TEMPERATURE, remote=True)

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs: Any) -> PromptValue:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps", '')
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs['agent_scratchpad'] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return formatted


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        final_answer = re.search(r"Final Answer:(.*?)(?:\nQuestion:|$)", llm_output, re.DOTALL)
        if final_answer:
            answer = final_answer.group(1).strip()
            return AgentFinish(return_values={"output": answer}, log=llm_output)
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            print("Could not parse LLM output")
            action = re.search(r"Action\s*\d*\s*:(.*?)$", llm_output, re.DOTALL)
            if not match:
                print("Couldn't even find an action, giving up.")
                return AgentFinish(return_values={"output": llm_output}, log=llm_output)
        action = match.group(1).strip()
        if match:
            action_input = match.group(2)
        else:
            action_input = ''
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class MyCustomAsyncHandler(AsyncCallbackHandler):
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token)
        self.callback(token)

def get_llm():
    if os.environ.get('OPENAI_API_TYPE') == 'openai':
        llm = OpenAI(temperature=TEMPERATURE)
    elif os.environ.get('OPENAI_API_TYPE') == 'azure':
        llm = AzureOpenAI(temperature=TEMPERATURE, deployment_name=os.environ.get('OPENAI_DEPLOYMENT_NAME'))
    else:
        raise KeyError("No OPENAI_API_TYPE environment variable set or invalid value")
    return llm

class Agent:
    def __init__(self, character: str):
        memory = MotorheadMemory(session_id="static", memory_key="history", url="http://motorhead:8080")
        memory.init()
        # memory = ConversationBufferWindowMemory(memory_key="history", k=20)
        tools = self._setup_tools()
        llm_chain = LLMChain(
            llm=get_llm(),
            prompt=self._setup_prompt_template(character, tools=tools))
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=CustomOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in tools])
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=memory)
        
    def _setup_tools(self) -> List[Tool]:
        # Define which tools the agent can use to answer user queries
        search = GoogleSearchAPIWrapper()
        wolfram = WolframAlphaAPIWrapper()
        transformers = TransformerToolAgent()
        coder = CodeAgent()
        codeExplainer = CodeExplainerAgent()
        tools = [
            Tool(name = "Search", func=search.run, description="useful for when you need to answer questions about current events"),
            Tool(name="Wolfram", func=wolfram.run, description="useful for when you need to answer factual questions about math, science, society, the time or culture"),
            Tool(name="Code", func=coder.run, description="useful for when you need to complete some code"),
            Tool(name="CodeExplainer", func=codeExplainer.run, description="useful for when you need to explain some code"),
            Tool(name="Transformers", func=transformers.run, description="useful for when you don't know how else to answer")
        ]
        return tools
        
    def _setup_prompt_template(self, character: str, tools: List[Tool]) -> CustomPromptTemplate:
        template_str = character + """
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}], if there is no action there should be a Final Answer
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, required if the Action is "None"

If the Action is None, there _must_ be a Final Answer.

{history}
Question: {input}
{agent_scratchpad}"""
        return CustomPromptTemplate(
            input_variables=['history', 'input', 'intermediate_steps'], 
            template=template_str,
            tools=tools)

    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None]) -> None:
        # await self.agent_executor.arun({'input': prompt, 'agent_scratchpad': '', 'history': ''}, callbacks=[MyCustomAsyncHandler(callback)])
        try:
            callback(self.agent_executor.run({'input': prompt, 'history': ''})) # TODO: Find a better way to do this
        except ConnectionResetError:
            # Reconnect here
            callback("Connection reset by peer")
        
    def prompt(self, prompt: str) -> str:
        return self.agent_executor.run(prompt)
    
"""
Example:

Question: What is the current ethereum price?
Thought: This is a question about current events, so I should use the Search tool.
Action: Search
Observation: The Ethereum price is $1,765.73, a change of -3.43% over the past 24 hours as of 5:00 p.m. The recent price action in Ethereum left the tokens market ...
Thought: I now know the final answer
Final Answer: The current Ethereum price is $1,765.73

End Example"""