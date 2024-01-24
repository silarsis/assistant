## Tools
from langchain.agents import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
# from models.tools import apify
from models.tools.prompt_template import PromptTemplate
from models.tools import web_requests
from models.tools.memory import Memory

# New semantic kernel setup
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AsyncAzureOpenAI
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import AsyncOpenAI
from semantic_kernel.planning.stepwise_planner import StepwisePlanner

# from models.plugins.ScrapeText import ScrapeTextPlugin
from models.plugins.WolframAlpha import WolframAlphaPlugin
from models.plugins.GoogleDocs import GoogleDocLoaderPlugin
from semantic_kernel.core_plugins import FileIOPlugin, MathPlugin, TextPlugin, TimePlugin

from typing import List, Optional, Callable
import os
import json


DEFAULT_SESSION_ID = 'static'

def getKernel(model: Optional[str] = "") -> sk.Kernel:
    kernel = sk.Kernel()
    deployment_name = os.environ.get('OPENAI_DEPLOYMENT_NAME', "")
    api_key = os.environ.get('OPENAI_API_KEY', "")
    endpoint = os.environ.get('OPENAI_API_BASE', "")
    org_id = os.environ.get('OPENAI_ORG_ID', None)
    model = model or os.environ.get('OPENAI_DEPLOYMENT_NAME', "gpt-4")
    # from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import AsyncOpenAI
    if os.environ.get('OPENAI_API_TYPE') == 'azure':
        print("Azure")
        client = AsyncAzureOpenAI(api_key=api_key, organization=org_id, base_url=endpoint)
        kernel.add_chat_service("echo", AzureChatCompletion(deployment_name, async_client=client))
    else:
        print("OpenAI")
        client = AsyncOpenAI(api_key=api_key, organization=org_id, base_url=endpoint)
        kernel.add_chat_service("echo", OpenAIChatCompletion(model, async_client=client))
    return kernel


class Guide:
    def __init__(self, default_character: str):
        print("Initialising Guide")
        self.guide = getKernel()
        self._setup_planner()
        self.memory = Memory(kernel=self.guide)
        self.default_character = default_character
        self.direct_responder = DirectResponse(self.guide, character=self.default_character)
        print("Guide initialised")
        
    def _setup_planner(self):
        print("Setting up the planner and plugins")
        # self.guide.import_plugin(ScrapeTextPlugin, "ScrapeText")
        self._google_docs = GoogleDocLoaderPlugin(kernel=self.guide)
        self.guide.import_plugin(self._google_docs, "gdoc")
        if os.environ.get('WOLFRAM_ALPHA_APPID'):
            self.guide.import_plugin(WolframAlphaPlugin(wolfram_alpha_appid=os.environ.get('WOLFRAM_ALPHA_APPID')), "wolfram")
        self.guide.import_plugin(MathPlugin(), "math")
        self.guide.import_plugin(FileIOPlugin(), "fileIO")
        self.guide.import_plugin(TimePlugin(), "time")
        self.guide.import_plugin(TextPlugin(), "text")
        self.planner = StepwisePlanner(self.guide)
        print("Planner created")
        
    def _setup_tools(self) -> List[Tool]:
        # Define which tools the agent can use to answer user queries
        print("Setting up tools")
        tools = []
        tools.append(Tool(name='Request', func=web_requests.scrape_text, description="use to make a request to a website, provide the url as action input"))
        # if os.environ.get('APIFY_API_TOKEN'):
        #     self.apify = apify.ApifyTool()
        #     tools.append(Tool(name='Scrape', func=self.apify.scrape_website, description="use when you need to scrape a website, provide the url as action input"))
        #     tools.append(Tool(name='Lookup', func=self.apify.query, description="use when you need to check if you already know something, provide the query as action input"))
        if os.environ.get('GOOGLE_API_KEY'):
            search = GoogleSearchAPIWrapper()
            tools.append(Tool(name="Search", func=search.run, description="use when you need to search for something on the internet"))
        print(f"Tools: {[tool.name for tool in tools]}")
        return tools
    
    async def _plan(self, goal: str, callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID) -> str:
        try:
            plan = self.planner.create_plan(goal=goal)
            context = self.guide.create_new_context()
            context.variables.set('session_id', session_id)
            response = await plan.invoke_async(context=context)
        except Exception as e:
            print(f"Planning failed: {e}")
            response = ""
        return str(response)
    
    async def rephrase(self, query: str, answer: str, history: str, history_context: str, session_id: str = DEFAULT_SESSION_ID) -> str:
        # Rephrase the text to match the character
        # TODO: Is there a way to force calling a particular plugin at the end of all other plugins in the planner?
        # If so, we could force rephrasing that way.
        # The rephrase question here sometimes generates an odd sort of response, should think about phrasing that better.
        return await self.direct_responder.response(history_context, history, f'You were asked "{query}" and you worked out the answer to be "{answer}". Please use this answer to respond to the user.', session_id=session_id)
    
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None], session_id: str=DEFAULT_SESSION_ID, **kwargs) -> None:
        # Convert the prompt to character + history
        history = self.memory.get_formatted_history(session_id=session_id)
        history_context = self.memory.get_context(session_id=session_id)
        self.memory.add_message(role="Human", content=f'Human: {prompt}', session_id=session_id)
        response = await self._plan(prompt, callback=callback)
        response = await self.rephrase(prompt, str(response), history, history_context, session_id=session_id)
        if not response:
            # If planning fails, try a chat
            response = await self.direct_responder.response(history_context, history, prompt, session_id=session_id)
        response = str(response)
        self.memory.add_message(role="AI", content=f"Response: {response}\n", session_id=session_id)
        return callback(response)

    async def update_prompt_template(self, prompt: str, callback: Callable[[str], None], **kwargs) -> str:
        self.direct_responder._prompt_templates.set(kwargs.get('session_id', DEFAULT_SESSION_ID), prompt)
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
        
    async def response(self, context: str, history: str, query: str, session_id: Optional[str] = DEFAULT_SESSION_ID, **kwargs) -> str:
        prompt = self._prompt_templates.get(session_id, self.kernel)
        ctx = self.kernel.create_new_context()
        ctx.variables['context'] = context
        ctx.variables['history'] = history
        ctx.variables['query'] = query
        response = await prompt.invoke_async(context=ctx)
        return str(response).strip()