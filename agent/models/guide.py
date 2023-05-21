from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from typing import List, Optional, Callable
from models.tools import web_requests
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


DEFAULT_PROMPT="""
{{character}}

Use the following format for your answers:

Human: the input question you must answer
Thought: you should always think about what to do, and check whether the answer is in the chat history or not
Criticism: you should always criticise your own actions
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action, or the answer if you are using the Answer tool

Example:

Human: what's the current price of ethereum?
Thought: I need to check the price of ethereum, I can do that by using the Search tool
Criticism: I should check the chat history first to see if I've already answered this question
Action: Search
Action Input: ethereum price

Context:
{{await 'context'}}

Chat History:
{{await 'history'}}

Human: {{await 'query'}}
Thought: {{gen 'thought'}}
Criticism: {{gen 'criticism'}}
Action: {{gen 'action'}}
Action Input: {{gen 'action_input' stop='Human:'}}
"""

class Memory:
    url: str = "http://motorhead:8080"
    timeout = 3000
    memory_key = "history"
    context: Optional[str] = None
    
    def __init__(self):
        self.context = {}
        self.messages = {}
    
    def refresh_from(self, session_id: str):
        res = requests.get(
            f"{self.url}/sessions/{session_id}/memory",
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )
        res_data = res.json()
        self.context[session_id] = res_data.get("context", "NONE")
        messages = res_data.get("messages", [])
        messages.reverse()
        self.messages[session_id] = messages # Not strictly thread safe, but not too harmful
        # print(f"Memory refreshed, current context: {self.context[session_id]}, length: {len(self.messages[session_id])}")

    def add_message(self, role: str, content: str, session_id: str):
        requests.post(
            f"{self.url}/sessions/{session_id}/memory",
            timeout=self.timeout,
            json={
                "messages": [
                    {"role": role, "content": f"{content}"},
                ]
            },
            headers={"Content-Type": "application/json"},
        )
        self.messages.setdefault(session_id, []).append({"role": role, "content": content})
        
    def get_history(self, session_id) -> List[str]:
        if session_id not in self.messages:
            self.refresh_from(session_id)
        return [message["content"] for message in self.messages[session_id]]
    
    def get_formatted_history(self, session_id) -> str:
        history = self.get_history(session_id)
        return "\n".join(history)
    
    def get_context(self, session_id) -> str:
        self.refresh_from(session_id)
        return self.context[session_id]


class PromptTemplate:
    def __init__(self, default_character: str, default_prompt: str):
        self.default_character = default_character
        self.default_prompt = default_prompt
        self._characters = {}
        self._prompt_template_str = {}
        self._prompt_templates = {}
        
    def get(self, session_id: str, llm) -> guidance.Program:
        if session_id in self._prompt_templates:
            return self._prompt_templates[session_id]
        character = self._characters.setdefault(session_id, self.default_character)
        template = self._prompt_template_str.setdefault(session_id, self.default_prompt)
        self._prompt_templates[session_id] = guidance(template, llm=llm, character=character)
        return self._prompt_templates[session_id]
    
    def set(self, session_id: str, prompt: str):
        self._prompt_template_str[session_id] = prompt
        self._prompt_templates.pop(session_id, None)
        
    def set_character(self, session_id: str, character: str):
        self._characters[session_id] = character
        self._prompt_templates.pop(session_id, None)


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
    def __init__(self, default_character: str):
        print("Initialising Guide")
        self.memory = Memory()
        self.tools = self._setup_tools()
        # self.guide = guidance.llms.transformers.Vicuna(load_vicuna())
        self.guide = guidance.llms.OpenAI('text-davinci-003')
        self.default_character = default_character
        self._prompt_templates = PromptTemplate(default_character, DEFAULT_PROMPT)
        print("Guide initialised")
        
    def _setup_tools(self) -> List[Tool]:
        # Define which tools the agent can use to answer user queries
        tools = []
        tools.append(Tool(name='Answer', func=lambda x: x, description="use when you already know the answer"))
        tools.append(Tool(name='Clarify', func=lambda x: x, description="use when you need more information"))
        tools.append(Tool(name='Request', func=web_requests.scrape_text, description="use when you need to make a request to a website, provide the url as action input"))
        # tools.append(WriteFileTool())
        # tools.append(ReadFileTool())
        if os.environ.get('WOLFRAM_ALPHA_APPID'):
            wolfram = WolframAlphaAPIWrapper()
            tools.append(Tool(name="Wolfram", func=wolfram.run, description="use when you need to answer factual questions about math, science, society, the time or culture"))
        if os.environ.get('GOOGLE_API_KEY'):
            search = GoogleSearchAPIWrapper()
            tools.append(Tool(name="Search", func=search.run, description="use when you need to answer questions about current events"))
        print(f"Tools: {[tool.name for tool in tools]}")
        return tools
        
    def _get_prompt_template(self, session_id: str) -> str:
        return self._prompt_templates.get(session_id, self.guide)(tool_names=[tool.name for tool in self.tools])
        
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None], **kwargs) -> None:
        response = self.prompt(query=prompt, interim=callback, hear_thoughts=kwargs.get('hear_thoughts', False), session_id=kwargs.get('session_id', 'static'))
        return callback(response)
    
    def prompt(self, query: str, history: str="", interim: Optional[Callable[[str], None]]=None, **kwargs) -> str:
        print(f"Prompt: {query}")
        session_id = kwargs.get('session_id', 'static')
        hear_thoughts = kwargs.get('hear_thoughts', False)
        if not history:
            history = self.memory.get_formatted_history(session_id=session_id)
        # print(f"History: {history}")
        history_context = self.memory.get_context(session_id=session_id)
        self.memory.add_message(role="Human", content=f'Human: {query}', session_id=session_id)
        response = self._get_prompt_template(session_id)(query=query, history=history, context=history_context)
        action = response['action'].strip()
        action_input = response['action_input'].strip()
        self.memory.add_message(role="AI", content=f"Action: {action}\nAction Input: {action_input}\n", session_id=session_id)
        # Clarify should probably actually do something interesting with the history or something
        if action in ('Answer', 'Clarify'):
            # This represents a completed answer
            return action_input
        print(f"Looking for tool for action '{action}'")
        if interim and hear_thoughts:
            interim(f"Thoughts: {response['thought']}.\nAction: {action}.\nAction Input: {action_input}.\n")
        tool = next((tool for tool in self.tools if tool.name.lower() == action.lower()), None)
        if tool:
            # Call the tool, include the output into the history and then recall the prompt
            print(f"  Calling {tool.name} with input {action_input}")
            try:
                tool_output = tool.func(action_input)
            except:
                print("  tool raised an exception")
                tool_output = "This tool failed to run"
            self.memory.add_message(role="AI", content=f"Outcome: {tool_output}", session_id=session_id)
            return self.prompt(query=query)
        else:
            print(f"  No tool found for action '{action}'")
            return self.prompt(query=query, history=f"{self.memory.get_context(session_id=session_id)}\nAction: {action}\nAction Input: {action_input}\nOutcome: No tool found for action '{action}'\n")

    async def update_prompt_template(self, prompt: str, callback: Callable[[str], None], **kwargs) -> None:
        self._prompt_templates.set(kwargs.get('session_id', 'static'), prompt)