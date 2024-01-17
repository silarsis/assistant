## Tools
from langchain.agents import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
# from models.tools import apify
from models.tools.prompt_template import PromptTemplate
from models.tools import web_requests
from models.tools.memory import Memory

# New semantic kernel setup
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from semantic_kernel.planning.action_planner import ActionPlanner
# from models.plugins.ScrapeText import ScrapeTextSkill
from models.plugins.WolframAlpha import WolframAlphaPlugin
from models.plugins.GoogleDocs import GoogleDocLoaderPlugin
from semantic_kernel.core_skills import FileIOSkill, MathSkill, TextSkill, TimeSkill

from typing import List, Optional, Callable
import os
import json


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
        # self.guide.import_skill(ScrapeTextSkill, "ScrapeText")
        self._google_docs = GoogleDocLoaderPlugin(kernel=self.guide)
        self.guide.import_skill(self._google_docs, "gdoc")
        if os.environ.get('WOLFRAM_ALPHA_APPID'):
            self.guide.import_skill(WolframAlphaPlugin(wolfram_alpha_appid=os.environ.get('WOLFRAM_ALPHA_APPID')), "wolfram")
        self.guide.import_skill(MathSkill(), "math")
        self.guide.import_skill(FileIOSkill(), "fileIO")
        self.guide.import_skill(TimeSkill(), "time")
        self.guide.import_skill(TextSkill(), "text")
        self.planner = ActionPlanner(self.guide)
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
    
    async def _plan(self, goal: str) -> str:
        try:
            plan = await self.planner.create_plan_async(goal=goal)
            response = await plan.invoke_async()
        except Exception as e:
            print(f"Planning failed: {e}")
            response = ""
        return str(response)
    
    async def rephrase(self, query: str, answer: str, history: str, history_context: str, session_id: str = 'static') -> str:
        # Rephrase the text to match the character
        # TODO: Is there a way to force calling a particular skill at the end of all other skills in the planner?
        # If so, we could force rephrasing that way.
        return await self.direct_responder.response(history_context, history, f'Please rephrase the answer "{answer}" to the query "{query}" according to your character', session_id=session_id)
    
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None], session_id: str='static', **kwargs) -> None:
        # Convert the prompt to character + history
        history = self.memory.get_formatted_history(session_id=session_id)
        history_context = self.memory.get_context(session_id=session_id)
        self.memory.add_message(role="Human", content=f'Human: {prompt}', session_id=session_id)
        response = await self._plan(prompt)
        response = await self.rephrase(prompt, str(response), history, history_context, session_id=session_id)
        if not response:
            # If planning fails, try a chat
            response = await self.direct_responder.response(history_context, history, prompt, session_id=session_id)
        response = str(response)
        self.memory.add_message(role="AI", content=f"Response: {response}\n", session_id=session_id)
        return callback(response)

    async def update_prompt_template(self, prompt: str, callback: Callable[[str], None], **kwargs) -> str:
        self.direct_responder._prompt_templates.set(kwargs.get('session_id', 'static'), prompt)
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
        
    async def response(self, context: str, history: str, query: str, session_id: Optional[str] = 'static', **kwargs) -> str:
        prompt = self._prompt_templates.get(session_id, self.kernel)
        ctx = self.kernel.create_new_context()
        ctx.variables['context'] = context
        ctx.variables['history'] = history
        ctx.variables['query'] = query
        response = await prompt.invoke_async(context=ctx)
        return str(response).strip()