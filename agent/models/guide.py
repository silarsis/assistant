## Tools
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from models.tools import apify
from models.tools import web_requests
from langchain.document_loaders import TextLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

from huggingface_hub import hf_hub_download, try_to_load_from_cache
from typing import List, Optional, Callable
import guidance
import requests
import os

# /openai/deployments/text-davinci-003/chat/completions?api-version=2023-03-15-preview

## Monkey patch
from guidance.llms import _openai
from guidance.llms._llm import LLMSession
import asyncio
import openai
import types

def add_text_to_chat_mode(chat_mode):
    if isinstance(chat_mode, types.GeneratorType):
        return _openai.add_text_to_chat_mode_generator(chat_mode)
    else:
        for c in chat_mode['choices']:
            c['text'] = c['message']['content'] or 'None'
        return chat_mode

_openai.add_text_to_chat_mode = add_text_to_chat_mode

class OpenAISession(LLMSession):
    async def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None, top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=None, cache_seed=0, caching=None):
        """ Generate a completion of the given prompt.
        """

        # we need to stream in order to support stop_regex
        if stream is None:
            stream = stop_regex is not None
        assert stop_regex is None or stream, "We can only support stop_regex for the OpenAI API when stream=True!"
        assert stop_regex is None or n == 1, "We don't yet support stop_regex combined with n > 1 with the OpenAI API!"

        assert token_healing is None or token_healing is False, "The OpenAI API does not yet support token healing! Please either switch to an endpoint that does, or don't use the `token_healing` argument to `gen`."

        # set defaults
        if temperature is None:
            temperature = self.llm.temperature

        # get the arguments as dictionary for cache key generation
        args = locals().copy()

        assert not pattern, "The OpenAI API does not support Guidance pattern controls! Please either switch to an endpoint that does, or don't use the `pattern` argument to `gen`."
        # assert not stop_regex, "The OpenAI API does not support Guidance stop_regex controls! Please either switch to an endpoint that does, or don't use the `stop_regex` argument to `gen`."

        # define the key for the cache
        key = self._cache_key(args)
        
        # allow streaming to use non-streaming cache (the reverse is not true)
        if key not in self.llm.__class__.cache and stream:
            args["stream"] = False
            key1 = self._cache_key(args)
            if key1 in self.llm.__class__.cache:
                key = key1
        
        # check the cache
        if key not in self.llm.__class__.cache or (caching is not True and not self.llm.caching) or caching is False:

            # ensure we don't exceed the rate limit
            while self.llm.count_calls() > self.llm.max_calls_per_min:
                await asyncio.sleep(1)

            fail_count = 0
            while True:
                try_again = False
                try:
                    self.llm.add_call()
                    call_args = {
                        "model": self.llm.model_name,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "n": n,
                        "stop": stop,
                        "logprobs": logprobs,
                        "echo": echo,
                        "stream": stream,
                        "engine": self.llm.model_name,
                    }
                    if logit_bias is not None:
                        call_args["logit_bias"] = {str(k): v for k,v in logit_bias.items()} # convert keys to strings since that's the open ai api's format
                    out = self.llm.caller(**call_args)

                except openai.error.RateLimitError:
                    await asyncio.sleep(3)
                    try_again = True
                    fail_count += 1
                
                if not try_again:
                    break

                if fail_count > self.llm.max_retries:
                    raise Exception(f"Too many (more than {self.llm.max_retries}) OpenAI API RateLimitError's in a row!")

            if stream:
                return self.llm.stream_then_save(out, key, stop_regex, n)
            else:
                self.llm.__class__.cache[key] = out
        
        # wrap as a list if needed
        if stream:
            if isinstance(self.llm.__class__.cache[key], list):
                return self.llm.__class__.cache[key]
            return [self.llm.__class__.cache[key]]
        
        return self.llm.__class__.cache[key]

guidance.llms._openai.OpenAISession = OpenAISession
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

DEFAULT_TOOL_PROMPT = """
{{character}}

Given the following information:

{{tool_output}}

The answer to the question '{{query}}' is:

{{gen 'answer'}}
"""

class LocalMemory:
    context: Optional[str] = None
    
    def __init__(self, llm=None, default_character=None):
        self.context = {}
        self.messages = {}
        self.llm = llm
        self.default_character = default_character
        self._prompt_templates = PromptTemplate(default_character, "Summarise the following conversation, taking the existing context into account:\nContext:\n{{context}}\n\nHistory:{{history}}\n\nSummary:{{gen 'summary'}}")
        
    def refresh_from(self, session_id: str):
        self.context.setdefault(session_id, "")
        self.messages.setdefault(session_id, [])
        return
    
    def _summarise(self, session_id: str):
        contextualise = self.messages[session_id][:-10]
        if not contextualise:
            return
        self.messages[session_id] = self.messages[session_id][-10:]
        prompt = self._prompt_templates.get(session_id, self.llm)
        response = prompt(context=self.get_context(), history="\n".join([message["content"] for message in contextualise]))
        print(response['summary'])
        return response['summary']
    
    def add_message(self, role: str, content: str, session_id: str):
        self.messages.setdefault(session_id, []).append({"role": role, "content": content})
        if len(self.messages[session_id]) > 20:
            self._summarise(session_id)
        self.context[session_id] = content
        
    def get_history(self, session_id) -> List[str]:
        return [message["content"] for message in self.messages.setdefault(session_id, [])]

    def get_formatted_history(self, session_id) -> str:
        history = self.get_history(session_id)
        return "\n".join(history)
    
    def get_context(self, session_id) -> str:
        return self.context.setdefault(session_id, "")

class MotorheadMemory:
    url: str = "http://motorhead:8080"
    timeout = 3000
    memory_key = "history"
    context: Optional[str] = None
    
    def __init__(self, llm=None, default_character=None):
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

if os.environ.get('MEMORY') == 'local':
    Memory = LocalMemory
else:
    Memory = MotorheadMemory

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
        self._google_docs_tokens = {}
        self.tools = self._setup_tools()
        # self.guide = guidance.llms.transformers.Vicuna(load_vicuna())
        self.guide = guidance.llms.OpenAI('text-davinci-003')
        self.memory = Memory(llm=self.guide, default_character=default_character)
        self.default_character = default_character
        self._prompt_templates = PromptTemplate(default_character, DEFAULT_PROMPT)
        self._tool_response_prompt_templates = PromptTemplate(default_character, DEFAULT_TOOL_PROMPT)
        print("Guide initialised")
        
    def _setup_tools(self) -> List[Tool]:
        # Define which tools the agent can use to answer user queries
        tools = []
        tools.append(Tool(name='Answer', func=lambda x: x, description="use when you already know the answer"))
        tools.append(Tool(name='Clarify', func=lambda x: x, description="use when you need more information"))
        tools.append(Tool(name='Request', func=web_requests.scrape_text, description="use when you need to make a request to a website, provide the url as action input"))
        # tools.append(WriteFileTool())
        # tools.append(ReadFileTool())
        if os.environ.get('APIFY_API_TOKEN'):
            self.apify = apify.ApifyTool()
            tools.append(Tool(name='Scrape', func=self.apify.scrape_website, description="use when you need to scrape a website, provide the url as action input"))
            tools.append(Tool(name='Lookup', func=self.apify.query, description="use when you need to check if you already know something, provide the query as action input"))
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
    
    def _get_tool_response_prompt_template(self, session_id: str) -> str:
        return self._tool_response_prompt_templates.get(session_id, self.guide)
        
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None], **kwargs) -> None:
        response = self.prompt(query=prompt, interim=callback, hear_thoughts=kwargs.get('hear_thoughts', False), session_id=kwargs.get('session_id', 'static'))
        return callback(response)
    
    def prompt(self, query: str, history: str="", interim: Optional[Callable[[str], None]]=None, **kwargs) -> str:
        print(f"Prompt: {query}")
        session_id = kwargs.get('session_id', 'static')
        hear_thoughts = kwargs.get('hear_thoughts', False)
        if not history:
            history = self.memory.get_formatted_history(session_id=session_id)
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
            except Exception as e:
                print("  tool raised an exception")
                print(e)
                tool_output = "This tool failed to run"
            self.memory.add_message(role="AI", content=f"Outcome: {tool_output}", session_id=session_id)
            response = self._get_tool_response_prompt_template(session_id)(query=query, tool_output=tool_output)
            self.memory.add_message(role="AI", content=f"Action: Answer\nAction Input: {response['answer']}\n", session_id=session_id)
            return response['answer']
        else:
            print(f"  No tool found for action '{action}'")
            return self.prompt(query=query, history=f"{self.memory.get_context(session_id=session_id)}\nAction: {action}\nAction Input: {action_input}\nOutcome: No tool found for action '{action}'\n")

    async def update_prompt_template(self, prompt: str, callback: Callable[[str], None], **kwargs) -> str:
        self._prompt_templates.set(kwargs.get('session_id', 'static'), prompt)
        return "Done"
        
    def update_google_docs_token(self, token: str, **kwargs) -> str:
        self._google_docs_tokens[kwargs.get('session_id', 'static')] = token
        return "Done"