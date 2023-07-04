## Tools
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
# from langchain.tools.file_management.write import WriteFileTool
# from langchain.tools.file_management.read import ReadFileTool
from langchain import OpenAI
from models.tools import apify
from models.tools import web_requests
from models.tools.google_docs import GoogleDocLoader

from models.codeagent import AzureCodeAgentExplain

from huggingface_hub import hf_hub_download, try_to_load_from_cache
import tiktoken
from typing import List, Optional, Callable
import guidance
import requests
import os
import json
from pydantic import create_model

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

def getLLM(model: Optional[str] = None):
    model = os.environ.get('OPENAI_MODEL', 'text-davinci-003')
    if os.environ.get('OPENAI_API_TYPE') == 'azure':
        print("Azure")
        return guidance.llms.OpenAI(
            model,
            api_type=os.environ.get('OPENAI_API_TYPE'),
            api_key=os.environ.get('OPENAI_API_KEY'),
            api_base=os.environ.get('OPENAI_API_BASE'),
            api_version=os.environ.get('OPENAI_API_VERSION'),
            deployment_id=os.environ.get('OPENAI_DEPLOYMENT_NAME')
        )
    else:
        print("OpenAI")
        # TODO need to monkey patch guidance for tiktoken encoder selection
        tiktoken.model.MODEL_TO_ENCODING.setdefault(model, 'p50k_base') # This is a guess
        return guidance.llms.OpenAI(model)
    
DEFAULT_CHARACTER="""
You are an AI assistant.
Your name is Echo.
You are designed to be helpful, but you are also a bit of a smartass. You don't have to be polite.
Your goal is to provide the user with answers to their questions and sometimes make them laugh.
Use Markdown formatting in your answers where appropriate.
You should answer in a personalised way, not too formal, concise, and not always polite.
"""

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
Thought: {{gen 'thought' temperature=0.7}}
Criticism: {{gen 'criticism' temperature=0.7}}
Action: {{gen 'action' stop='Action Input:'}}
Action Input: {{gen 'action_input' stop='Human:'}}
"""

DEFAULT_TOOL_PROMPT = """
{{character}}

You called the {{tool}} tool to answer the query '{{query}}'.
The {{tool}} tool returned the following answer:

{{tool_output}}

Please reword this answer to match your character, and add any additional information you think is relevant.

{{gen 'answer'}}
"""

class LocalMemory:
    context: Optional[str] = None
    
    def __init__(self, llm=None, default_character=None):
        print("Local Memory")
        self.context = {}
        self.messages = {}
        self.llm = llm
        self.default_character = default_character
        self._prompt_templates = PromptTemplate(default_character, "Summarise the following conversation, taking the existing context into account:\nContext:\n{{context}}\n\nHistory:{{history}}\n\nSummary:{{gen 'summary'}}")
        
    def refresh_from(self, session_id: str) -> None:
        self.context.setdefault(session_id, "")
        self.messages.setdefault(session_id, [])
    
    def _summarise(self, session_id: str) -> str:
        contextualise = self.messages[session_id][:-10]
        if not contextualise:
            return
        self.messages[session_id] = self.messages[session_id][-10:]
        prompt = self._prompt_templates.get(session_id, self.llm)
        response = prompt(context=self.get_context(session_id), history="\n".join([message["content"] for message in contextualise]))
        print(response['summary'])
        return response['summary']
    
    def add_message(self, role: str, content: str, session_id: str) -> None:
        self.messages.setdefault(session_id, []).append({"role": role, "content": content})
        if len(self.messages[session_id]) > 20:
            self._summarise(session_id)
        self.context[session_id] = content
        
    def get_history(self, session_id) -> List[str]:
        return [message["content"] for message in self.messages.setdefault(session_id, [])]

    def get_formatted_history(self, session_id: str) -> str:
        history = self.get_history(session_id)
        return "\n".join(history)
    
    def get_context(self, session_id) -> str:
        return self.context.setdefault(session_id, "")

class MotorheadMemory:
    url: str = "http://motorhead:8001"
    timeout = 3000
    memory_key = "history"
    context: Optional[str] = None
    
    def __init__(self, llm=None, default_character=None):
        print("Motorhead memory")
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
        self.guide = getLLM()
        self._google_docs = GoogleDocLoader(llm=OpenAI(temperature=0.4))
        self.tools = self._setup_tools()
        # self.guide = guidance.llms.transformers.Vicuna(load_vicuna())
        self.memory = Memory(llm=self.guide, default_character=default_character)
        self.default_character = default_character
        self._prompt_templates = PromptTemplate(default_character, DEFAULT_PROMPT)
        self._tool_response_prompt_templates = PromptTemplate(default_character, DEFAULT_TOOL_PROMPT)
        self.tool_selector = ToolSelector(self.tools, self.guide)
        self.character_adder = AddCharacter(self.guide)
        self.direct_responder = DirectResponse(self.guide)
        print("Guide initialised")
        
    def _setup_tools(self) -> List[Tool]:
        # Define which tools the agent can use to answer user queries
        tools = []
        tools.append(Tool(name='Answer', func=lambda x: x, description="use when you already know the answer"))
        tools.append(Tool(name='Clarify', func=lambda x: x, description="use when you need more information"))
        tools.append(Tool(name='Request', func=web_requests.scrape_text, description="use to make a request to a website, provide the url as action input"))
        tools.append(Tool(name='ExplainCode', func=AzureCodeAgentExplain().run, description="use for any coding-related requests, including code generation and explanation"))
        # tools.append(WriteFileTool())
        # tools.append(ReadFileTool())
        tools.append(Tool(name='LoadDocument', func=self._google_docs.load_doc, description="use to load a document, provide the document id as action input", args_schema=create_model('LoadDocumentModel', tool_input='', session_id='')))
        if os.environ.get('APIFY_API_TOKEN'):
            self.apify = apify.ApifyTool()
            tools.append(Tool(name='Scrape', func=self.apify.scrape_website, description="use when you need to scrape a website, provide the url as action input"))
            tools.append(Tool(name='Lookup', func=self.apify.query, description="use when you need to check if you already know something, provide the query as action input"))
        if os.environ.get('WOLFRAM_ALPHA_APPID'):
            wolfram = WolframAlphaAPIWrapper()
            tools.append(Tool(name="Wolfram", func=wolfram.run, description="use when you need to answer factual questions about math, science, society, the time or culture"))
        if os.environ.get('GOOGLE_API_KEY'):
            search = GoogleSearchAPIWrapper()
            tools.append(Tool(name="Search", func=search.run, description="use when you need to search for something on the internet"))
        print(f"Tools: {[tool.name for tool in tools]}")
        return tools
        
    def _get_prompt_template(self, session_id: str) -> str:
        return self._prompt_templates.get(session_id, self.guide)(tool_names=[tool.name for tool in self.tools])
    
    def _get_tool_response_prompt_template(self, session_id: str) -> str:
        return self._tool_response_prompt_templates.get(session_id, self.guide)
        
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None], **kwargs) -> None:
        response = self.prompt(query=prompt, interim=callback, hear_thoughts=kwargs.get('hear_thoughts', False), session_id=kwargs.get('session_id', 'static'))
        return callback(response)
    
    def _call_tool(self, tool, action_input: str, session_id: str) -> str:
        print(f"  Calling {tool.name} with input {action_input}")
        kwargs = {}
        if tool.args_schema and 'session_id' in json.loads(tool.args_schema.schema_json())['properties']:
            kwargs['session_id'] = session_id
        try:
            tool_output = tool.func(action_input, **kwargs)
        except Exception as e:
            print("  tool raised an exception")
            print(e)
            tool_output = "This tool failed to run"
        print(f"Tool Output: {tool_output}\n")
        return tool_output
    
    def prompt(self, query: str, history: str="", interim: Optional[Callable[[str], None]]=None, **kwargs) -> str:
        session_id = kwargs.get('session_id', 'static')
        hear_thoughts = kwargs.get('hear_thoughts', False)
        if not history:
            history = self.memory.get_formatted_history(session_id=session_id)
        history_context = self.memory.get_context(session_id=session_id)
        self.memory.add_message(role="Human", content=f'Human: {query}', session_id=session_id)
        # Select the tool to use
        (action, action_input) = self.tool_selector.select(query=query, session_id=session_id)
        print(f"Action: {action}\nAction Input: {action_input}\n")
        # Clarify should probably actually do something interesting with the history or something
        if action in ('Answer', 'Clarify'):
            # This represents a completed answer
            (thought, response) = self.direct_responder.response(
                history_context, 
                history, 
                query, 
                session_id=session_id
            )
            self.memory.add_message(role="AI", content=f"Action: {action}\nAction Input: {response}\n", session_id=session_id)
            print(f"Thought: {thought}\nResponse: {response}\n")
            if interim and hear_thoughts:
                interim(f"\nThought: {thought}.\n")
            return response
        print(f"Looking for tool for action '{action}'")
        tool = next((tool for tool in self.tools if tool.name.lower() == action.lower()), None)
        if tool:
            tool_output = self._call_tool(tool, action_input, session_id)
            self.memory.add_message(role="AI", content=f"Outcome: {tool_output}", session_id=session_id)
            response = self._get_tool_response_prompt_template(session_id)(query=query, tool=tool.name, tool_output=tool_output)
            # reworded_response = self.character_adder.reword(query, response['answer'], session_id=session_id)
            self.memory.add_message(role="AI", content=f"Action: Answer\nAction Input: {response['answer']}\n", session_id=session_id)
            return response['answer']
        else:
            print(f"  No tool found for action '{action}'")
            return self.prompt(
                query=query, 
                history=f"{self.memory.get_context(session_id=session_id)}\nAction: {action}\nAction Input: {action_input}\nOutcome: No tool found for action '{action}'\n"
            ) # TODO Add character here

    async def update_prompt_template(self, prompt: str, callback: Callable[[str], None], **kwargs) -> str:
        self._prompt_templates.set(kwargs.get('session_id', 'static'), prompt)
        callback("Done")
        
    async def update_google_docs_token(self, token: str, callback: Callable[[str], None], session_id: str ='', **kwargs) -> str:
        self._google_docs.set_token(json.loads(token), session_id=session_id)
        callback("Authenticated")
    
class AddCharacter:
    " Designed to run over the results of any query and add character to the response "
    character = DEFAULT_CHARACTER
    prompt = """
{{character}}
    
You were asked the following question:
{{await 'query'}}

Please reword the following answer to make it clearer or more interesting:
{{await 'answer'}}

{{gen 'character_answer'}}
"""
    def __init__(self, llm):
        if llm:
            self.llm = llm
        else:
            self.llm = getLLM()
        self._prompt_templates = PromptTemplate(self.character, self.prompt)
        
    def reword(self, query: str, answer: str, **kwargs) -> str:
        session_id = kwargs.get('session_id', 'static')
        response = self._prompt_templates.get(session_id, self.llm)(query=query, answer=answer)
        return response['character_answer'].strip()
        
class ToolSelector:
    " Designed to decide how to answer a question "
    character = """
You are an AI assistant with a handful of tools at your disposal.
Your job is to select the most appropriate tool for the query.
"""

    prompt = """
Your available tools are: {{tools}}.
The query from the user is :
'''
{{await 'query'}}
'''

Please select the best tool from the list above by name only.
The best tool is: {{gen 'tool'}}

Now, given the tool selected, please provide the input to the tool.
The tool input is: {{gen 'tool_input'}}
"""
    def __init__(self, tools: list, llm):
        self.tools = '\n'.join([f"{tool.name} ({tool.description})\n" for tool in tools])
        if llm:
            self.llm = llm
        else:
            self.llm = getLLM()
        self._prompt_templates = PromptTemplate(self.character, self.prompt)
        
    def select(self, query: str, **kwargs) -> str:
        session_id = kwargs.get('session_id', 'static')
        response = self._prompt_templates.get(session_id, self.llm)(tools=self.tools, query=query)
        tool = response['tool'].strip()
        tool_input = response['tool_input'].strip()
        return (tool, tool_input)
    
class DirectResponse:
    " Designed to answer a question directly "
    character = DEFAULT_CHARACTER
    prompt = """
{{character}}

Use the following format for your answers:

Human: the input question you must answer
Thought: you should always think about what to do, and check whether the answer is in the chat history or not
Criticism: you should always criticise your own actions
Revised Thought: based on the criticism, any modified thoughts should be recorded here
Response: your response to the user's question

Context:
{{await 'context'}}

Chat History:
{{await 'history'}}

Human: {{await 'query'}}
Thought: {{gen 'thought' temperature=0.7}}
Criticism: {{gen 'criticism' temperature=0.7}}
Revised Thought: {{gen 'revised_thought' temperature=0.7}}
Response: {{gen 'response' stop='Human:'}}
"""

    def __init__(self, llm):
        if llm:
            self.llm = llm
        else:
            self.llm = getLLM()
        self._prompt_templates = PromptTemplate(self.character, self.prompt)
        
    def response(self, context: str, history: str, query: str, **kwargs) -> str:
        session_id = kwargs.get('session_id', 'static')
        response = self._prompt_templates.get(session_id, self.llm)(context=context, history=history, query=query)
        return (response['revised_thought'].strip(), response['response'].strip())