from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.chat_models import AzureChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory.chat_memory import BaseChatMemory, get_prompt_input_key
import re
import os

from typing import Callable
from .generic import ModelClass


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        print(llm_output, flush=True)
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output)
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            #raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            print(f"Could not parse LLM output: `{llm_output}`")
            action = re.search(r"Action\s*\d*\s*:(.*?)\n", llm_output, re.DOTALL)
            if not match:
                print(f"Couldn't even find an action, giving up")
                return AgentFinish(
                    return_values={"output": llm_output},
                    log=llm_output)
        action = match.group(1).strip()
        if match:
            action_input = match.group(2)
        else:
            action_input = ''
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    

class Agent:
    def __init__(self, character: str):
        tools = self._setup_tools()
        tool_names = [tool.name for tool in tools]
        prompt_template = self._setup_prompt_template(character, tools=tools)
        output_parser = CustomOutputParser()
        llm = AzureChatOpenAI(temperature=0, deployment_name=os.environ.get('OPENAI_DEPLOYMENT_NAME'))
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain, 
            output_parser=output_parser,
            stop=["\nObservation:"], 
            allowed_tools=tool_names
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools, 
            verbose=True)
        
    def _setup_tools(self) -> List[Tool]:
        # Define which tools the agent can use to answer user queries
        search = GoogleSearchAPIWrapper()
        wolfram = WolframAlphaAPIWrapper()
        tools = [
            Tool(name = "Search", func=search.run, description="useful for when you need to answer questions about current events"),
            Tool(name="Wolfram", func=wolfram.run, description="useful for when you need to answer factual questions about math, science, society or culture")
        ]
        return tools
        
    def _setup_prompt_template(self, character: str, tools: List[Tool]) -> CustomPromptTemplate:
        template_str = character + """
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

If you have no action, use your thoughts and observations as a Final Answer

Question: {input}
{agent_scratchpad}
"""
        return CustomPromptTemplate(
            input_variables=["input", "intermediate_steps"], 
            template=template_str,
            tools=tools)

    def _setup_memory(self):
        # Define your embedding model
        embeddings_model = OpenAIEmbeddings()
        # Initialize the vectorstore as empty
        import faiss

        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        self.memory = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None]) -> None:
        try:
            callback(self.agent_executor.run(prompt)) # TODO: Find a better way to do this
        except ConnectionResetError:
            callback("Connection reset by peer")
        
    def prompt(self, prompt: str) -> str:
        # Here, we want to collect the output and then think about whether to feed it
        # back into itself or not.
        return self.agent_executor.run(prompt)
    
class AgentMemory(BaseChatMemory):
    retriever: VectorStoreRetriever = Field(exclude=True)
    """VectorStoreRetriever object to connect to."""

    @property
    def memory_variables(self) -> List[str]:
        return ["chat_history", "relevant_context"]

    def _get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
        """Get the input key for the prompt."""
        if self.input_key is None:
            return get_prompt_input_key(inputs, self.memory_variables)
        return self.input_key

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_key = self._get_prompt_input_key(inputs)
        query = inputs[input_key]
        docs = self.retriever.get_relevant_documents(query)
        return {
            "chat_history": self.chat_memory.messages[-10:],
            "relevant_context": docs,
        }