## Tools
from langchain.agents import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
# from langchain.tools.file_management.write import WriteFileTool
# from langchain.tools.file_management.read import ReadFileTool
from langchain_openai import OpenAI
# from models.tools import apify
from models.tools.prompt_template import PromptTemplate
from models.tools import web_requests
from models.tools.google_docs import GoogleDocLoader
from models.tools.planner import Planner
from models.tools.memory import Memory

# New semantic kernel setup
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from semantic_kernel.planning.action_planner import ActionPlanner
from models.plugins.ScrapeText import ScrapeTextSkill

from typing import List, Optional, Callable
import os
import json
from pydantic import create_model


def getKernel(model: Optional[str] = "") -> sk.Kernel:
    kernel = sk.Kernel()
    deployment = os.environ.get('OPENAI_DEPLOYMENT_NAME', "")
    api_key = os.environ.get('OPENAI_API_KEY', "")
    endpoint = os.environ.get('OPENAI_API_BASE', "")
    org_id = os.environ.get('OPENAI_ORG_ID', None)
    model = model or os.environ.get('OPENAI_DEPLOYMENT_NAME', "gpt-4")
    if os.environ.get('OPENAI_API_TYPE') == 'azure':
        print("Azure")
        kernel.add_chat_service("dv", AzureChatCompletion(deployment, endpoint, api_key=api_key))
    else:
        print("OpenAI")
        # api_key, org_id = sk.openai_settings_from_dot_env()
        if endpoint:
            # Need to use a different connector here to connect to the custom endpoint
            from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import AsyncOpenAI, OpenAIChatCompletion
            # create the openai connection
            client = AsyncOpenAI(api_key=api_key, organization=org_id, base_url=endpoint)
            kernel.add_chat_service("chat-gpt", OpenAIChatCompletion(model, async_client=client))
        else:
            from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
            kernel.add_chat_service("chat-gpt", OpenAIChatCompletion(model, api_key, org_id))
    return kernel

DEFAULT_PROMPT="""

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

You called the {{tool}} tool to answer the query '{{query}}'.
The {{tool}} tool returned the following answer:

{{tool_output}}

Please reword this answer to match your character, and add any additional information you think is relevant.

{{gen 'answer'}}
"""


class Guide:
    def __init__(self, default_character: str):
        print("Initialising Guide")
        self.guide = getKernel()
        self._google_docs = GoogleDocLoader(llm=OpenAI(temperature=0.4))
        self._setup_planner()
        self.memory = Memory(kernel=self.guide)
        self.default_character = default_character
        self._prompt_templates = PromptTemplate(default_character, DEFAULT_PROMPT)
        self.direct_responder = DirectResponse(self.guide, character=self.default_character)
        print("Guide initialised")
        
    def _setup_planner(self):
        print("Setting up the planner and plugins")
        self.guide.import_skill(ScrapeTextSkill, "ScrapeText")
        self.planner = ActionPlanner(self.guide)
        print("Planner created")
        
    def _setup_tools(self) -> List[Tool]:
        # Define which tools the agent can use to answer user queries
        print("Setting up tools")
        tools = []
        tools.append(Tool(name='Answer', func=lambda x: x, description="use when you already know the answer"))
        tools.append(Tool(name='Clarify', func=lambda x: x, description="use when you need more information"))
        tools.append(Tool(name='Plan', func=lambda x: Planner(self.guide).run(self.tools, x), description="use when the request is going to take multiple steps and/or tools to complete"))
        tools.append(Tool(name='Request', func=web_requests.scrape_text, description="use to make a request to a website, provide the url as action input"))
        # tools.append(WriteFileTool())
        # tools.append(ReadFileTool())
        tools.append(Tool(name='LoadDocument', func=self._google_docs.load_doc, description="use to load a document, provide the document id as action input", args_schema=create_model('LoadDocumentModel', tool_input='', session_id='')))
        # if os.environ.get('APIFY_API_TOKEN'):
        #     self.apify = apify.ApifyTool()
        #     tools.append(Tool(name='Scrape', func=self.apify.scrape_website, description="use when you need to scrape a website, provide the url as action input"))
        #     tools.append(Tool(name='Lookup', func=self.apify.query, description="use when you need to check if you already know something, provide the query as action input"))
        if os.environ.get('WOLFRAM_ALPHA_APPID'):
            wolfram = WolframAlphaAPIWrapper()
            tools.append(Tool(name="Wolfram", func=wolfram.run, description="use when you need to answer factual questions about math, science, society, the time or culture"))
        if os.environ.get('GOOGLE_API_KEY'):
            search = GoogleSearchAPIWrapper()
            tools.append(Tool(name="Search", func=search.run, description="use when you need to search for something on the internet"))
        print(f"Tools: {[tool.name for tool in tools]}")
        return tools
    
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None], **kwargs) -> None:
        response = await self.prompt(query=prompt, interim=callback, hear_thoughts=kwargs.get('hear_thoughts', False), session_id=kwargs.get('session_id', 'static'))
        return callback(response)
    
    async def prompt(self, query: str, history: str="", interim: Optional[Callable[[str], None]]=None, session_id: str='static', hear_thoughts: bool=False, **kwargs) -> str:
        if not history:
            history = self.memory.get_formatted_history(session_id=session_id)
        history_context = self.memory.get_context(session_id=session_id)
        self.memory.add_message(role="Human", content=f'Human: {query}', session_id=session_id)
        response = await self.direct_responder.response(
            history_context, 
            history, 
            query, 
            session_id=session_id
        )
        self.memory.add_message(role="AI", content=f"Response: {response}\n", session_id=session_id)
        print(f"Response: {response}\n")
        return response

    async def update_prompt_template(self, prompt: str, callback: Callable[[str], None], **kwargs) -> str:
        self._prompt_templates.set(kwargs.get('session_id', 'static'), prompt)
        callback("Done")
        
    async def update_google_docs_token(self, token: str, callback: Callable[[str], None], session_id: str ='', **kwargs) -> str:
        self._google_docs.set_token(json.loads(token), session_id=session_id)
        callback("Authenticated")
    
class DirectResponse:
    " Designed to answer a question directly "
    prompt = """

Use the following format for your answers:

Human: the input question you must answer
Answer: your answer to the question

Context:
{{$context}}

Chat History:
{{$history}}

Human: {{$query}}
Answer: """

    def __init__(self, kernel: sk.Kernel, character: str = ""):
        self.kernel = kernel
        self.character = character
        self._prompt_templates = PromptTemplate(self.character, self.prompt)
        
    async def response(self, context: str, history: str, query: str, **kwargs) -> str:
        session_id = kwargs.get('session_id', 'static')
        prompt = self._prompt_templates.get(session_id, self.kernel)
        ctx = self.kernel.create_new_context()
        ctx.variables['context'] = context
        ctx.variables['history'] = history
        ctx.variables['query'] = query
        print(ctx.variables)
        response = await prompt.invoke_async(context=ctx)
        return str(response).strip()