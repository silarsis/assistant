from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from typing import List, Optional, Callable
import guidance
import requests
import os

## Monkey patch
from guidance.llms import _openai
import types

def add_text_to_chat_mode(chat_mode):
    if isinstance(chat_mode, types.GeneratorType):
        return _openai.add_text_to_chat_mode_generator(chat_mode)
    else:
        for c in chat_mode['choices']:
            c['text'] = c['message']['content'] or 'None'
        return chat_mode

_openai.add_text_to_chat_mode = add_text_to_chat_mode
## End monkey patch


TEMPERATURE = 0.2


class Memory:
    url: str = "http://motorhead:8080"
    timeout = 3000
    memory_key = "history"
    session_id: str
    context: Optional[str] = None
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.refresh_from()
        
    def refresh_from(self):
        res = requests.get(
            f"{self.url}/sessions/{self.session_id}/memory",
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )
        res_data = res.json()
        self.context = res_data.get("context", "NONE")
        messages = res_data.get("messages", [])
        messages.reverse()
        self.messages = messages # Not strictly thread safe, but not too harmful
        print(f"Memory refreshed, current context: {self.context}, length: {len(self.messages)}")

    def add_message(self, role: str, content: str):
        requests.post(
            f"{self.url}/sessions/{self.session_id}/memory",
            timeout=self.timeout,
            json={
                "messages": [
                    {"role": role, "content": f"{content}"},
                ]
            },
            headers={"Content-Type": "application/json"},
        )
        self.messages.append({"role": role, "content": content})
        
    def get_history(self) -> List[str]:
        return [message["content"] for message in self.messages]
    
    def get_context(self) -> str:
        self.refresh_from()
        return self.context


def load_vicuna():
    print("Loading Vicuna")
    filename = hf_hub_download(
        repo_id="TheBloke/WizardLM-13B-Uncensored-GGML", 
        filename="wizardLM-13B-Uncensored.ggml.q4_1.bin", 
        cache_dir="/models")
    print("Model download complete", flush=True)
    # Load the model
    filename = try_to_load_from_cache(
        repo_id="TheBloke/WizardLM-13B-Uncensored-GGML", 
        filename="wizardLM-13B-Uncensored.ggml.q4_1.bin", 
        cache_dir="/models")
    print(filename)
    return filename

class Guide:
    def __init__(self, character: str):
        print("Initialising Guide")
        self.memory = Memory(session_id="static")
        self.tools = self._setup_tools()
        self.guide = guidance.llms.transformers.Vicuna(load_vicuna())
        #self.guide = guidance.llms.OpenAI('text-davinci-003')
        self._prompt_template = self._setup_prompt_template(character)
        print("Guide initialised")
        
    def _setup_tools(self) -> List[Tool]:
        # Define which tools the agent can use to answer user queries
        tools = []
        tools.append(Tool(name='Answer', func=lambda x: x, description="use when you already know the answer"))
        tools.append(Tool(name='Clarify', func=lambda x: x, description="use when you need more information"))
        if os.environ.get('WOLFRAM_ALPHA_APPID'):
            wolfram = WolframAlphaAPIWrapper()
            tools.append(Tool(name="Wolfram", func=wolfram.run, description="use when you need to answer factual questions about math, science, society, the time or culture"))
        if os.environ.get('GOOGLE_API_KEY'):
            search = GoogleSearchAPIWrapper()
            tools.append(Tool(name="Search", func=search.run, description="use when you need to answer questions about current events"))
        return tools
        
    def _setup_prompt_template(self, character: str):
        return guidance("""
{{character}}

Use the following format for your answers:

Question: the input question you must answer
Thought: you should always think about what to do, and check whether the answer is in the chat history or not
Criticism: you should always criticise your own actions
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action, or the answer if you are using the Answer tool

History of chat so far:
{{await 'history'}}
End of History

Question: {{await 'query'}}
Thought: {{gen 'thought'}}
Criticism: {{gen 'criticism'}}
Action: {{gen 'action'}}
Action Input: {{gen 'action_input' stop='Question:'}}
""", llm=self.guide, character=character, tool_names=[tool.name for tool in self.tools])

    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None], hear_thoughts: bool = False) -> None:
        response = self.prompt(query=prompt, interim=callback, hear_thoughts=hear_thoughts)
        return callback(response)
    
    def prompt(self, query: str, history: str="", interim: Optional[Callable[[str], None]]=None, hear_thoughts: bool = False) -> str:
        print(f"Prompt: {query}")
        if not history:
            history = self.memory.get_history()
        print(f"History: {history}")
        self.memory.add_message(role="Human", content=f'Question: {query}')
        response = self._prompt_template(query=query, history=history)
        action = response['action'].strip()
        action_input = response['action_input'].strip()
        self.memory.add_message(role="AI", content=f"Action: {action}\nAction Input: {action_input}\n")
        # Clarify should probably actually do something interesting with the history or something
        if action in ('Answer', 'Clarify'):
            # This represents a completed answer
            return action_input
        print(f"Looking for tool for action '{action}'")
        if interim and hear_thoughts:
            interim(f"Thoughts: {response['thought']}\nAction: {action}\nAction Input: {action_input}\n")
        tool = next((tool for tool in self.tools if tool.name == action), None)
        if tool:
            # Call the tool, include the output into the history and then recall the prompt
            print(f"  Calling {tool.name} with input {action_input}")
            try:
                tool_output = tool.func(action_input)
            except:
                print("  tool raised an exception")
                tool_output = "This tool failed to run"
            self.memory.add_message(role="AI", content=f"Outcome: {tool_output}")
            return self.prompt(query=query, history=f"{self.memory.get_context()}\nAction: {action}\nAction Input: {action_input}\nOutcome: {tool_output}\n")
        else:
            print(f"  No tool found for action '{action}'")
            return self.prompt(query=query, history=f"{self.memory.get_context()}\nAction: {action}\nAction Input: {action_input}\nOutcome: No tool found for action '{action}'\n")
