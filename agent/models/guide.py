## Tools
import base64
from typing import List, Optional, Callable, Any, Literal
import os
from dotenv import load_dotenv

from pydantic import BaseModel

from langchain.agents import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper

# from models.tools import apify
from models.tools.prompt_template import PromptTemplate
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
from models.plugins.GoogleSearch import GoogleSearchPlugin
from models.plugins.ImageGeneration import ImageGenerationPlugin
from models.plugins.ScrapeText import ScrapeTextPlugin
from semantic_kernel.core_plugins import FileIOPlugin, MathPlugin, TextPlugin, TimePlugin, TextMemoryPlugin

DEFAULT_SESSION_ID = "static"


class Message(BaseModel):
    mesg: str = ""
    type: Literal["request", "response", "thought", "error"] = "request"
    
    def __str__(self):
        return self.model_dump_json()
    
    def __init__(self, *args, **kwargs):
        if args:
            kwargs['mesg'] = args[0] # Allow for init without specifying 'mesg='
        super().__init__(**kwargs)
    
class Response(Message):
    type: str = "response"
    
class Thought(Message):
    type: str = "thought"

def getKernel(model: Optional[str] = "") -> sk.Kernel:
    load_dotenv()
    kernel = sk.Kernel()
    deployment_name = os.environ.get("OPENAI_DEPLOYMENT_NAME", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    endpoint = os.environ.get("OPENAI_API_BASE", "")
    org_id = os.environ.get("OPENAI_ORG_ID", None)
    model = model or os.environ.get("OPENAI_DEPLOYMENT_NAME", "gpt-4")
    if os.environ.get("OPENAI_API_TYPE") == "azure":
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
        if os.environ.get("WOLFRAM_ALPHA_APPID"):
            self.guide.import_plugin(WolframAlphaPlugin(wolfram_alpha_appid=os.environ.get("WOLFRAM_ALPHA_APPID")), "wolfram")
        self.guide.import_plugin(MathPlugin(), "math")
        self.guide.import_plugin(FileIOPlugin(), "fileIO")
        self.guide.import_plugin(TimePlugin(), "time")
        self.guide.import_plugin(TextPlugin(), "text")
        self.guide.import_plugin(TextMemoryPlugin(), "text_memory")
        self.guide.import_plugin(ImageGenerationPlugin(), "image generation")
        self.guide.import_plugin(ScrapeTextPlugin(), "scrape_text")
        self.guide.import_plugin(GoogleSearchPlugin(), "google_search")
        self.planner = StepwisePlanner(self.guide)
        print("Planner created")

    def _setup_tools(self) -> List[Tool]:
        # Define which tools the agent can use to answer user queries
        print("Setting up tools")
        tools = []
        # if os.environ.get('APIFY_API_TOKEN'):
        #     self.apify = apify.ApifyTool()
        #     tools.append(Tool(name='Scrape', func=self.apify.scrape_website, description="use when you need to scrape a website, provide the url as action input"))
        #     tools.append(Tool(name='Lookup', func=self.apify.query, description="use when you need to check if you already know something, provide the query as action input"))
        if os.environ.get("GOOGLE_API_KEY"):
            search = GoogleSearchAPIWrapper()
            tools.append(Tool(name="Search", func=search.run, description="use when you need to search for something on the internet"))
        print(f"Tools: {[tool.name for tool in tools]}")
        return tools

    async def _plan(self, goal: str, callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False) -> Message:
        try:
            plan = self.planner.create_plan(goal=goal)
            if hear_thoughts:
                pass
                #callback(Thought(mesg=f"Planning result"))
            context = self.guide.create_new_context()
            context.variables.set("session_id", session_id)
            result = await plan.invoke_async(context=context)
        except Exception as e:
            print(f"Planning failed: {e}")
            if hear_thoughts:
                callback(Thought(mesg=str(e)))
            result = ""
        return Response(mesg=str(result))

    async def rephrase(self, input: str, answer: Message, history: str, history_context: str, session_id: str = DEFAULT_SESSION_ID) -> Message:
        # Rephrase the text to match the character
        # TODO: Is there a way to force calling a particular plugin at the end of all other plugins in the planner?
        # If so, we could force rephrasing that way.
        prompt = PromptTemplate(
            self.default_character, 
            f'You were asked "{input}" and you worked out the answer to be "{answer.mesg}". Please respond to the user.'
        ).get(session_id, self.guide)
        ctx = self.guide.create_new_context()
        ctx.variables['input'] = answer.mesg
        result = await prompt.invoke_async(context=ctx)
        return Response(mesg=str(result).strip())

    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False, **kwargs) -> Message:
        # Convert the prompt to character + history
        history = self.memory.get_formatted_history(session_id=session_id)
        history_context = self.memory.get_context(session_id=session_id)
        self.memory.add_message(role="Human", content=f"Human: {prompt}", session_id=session_id)
        response = await self._plan(prompt, callback=callback, hear_thoughts=hear_thoughts)
        if response:
            if hear_thoughts:
                callback(Thought(mesg=f"Rephrasing the answer from plugins: {response}"))
            response = await self.rephrase(prompt, response, history, history_context, session_id=session_id)
        else:
            # If planning fails, try a chat
            response = await self.direct_responder.response(history_context, history, prompt, session_id=session_id)
        self.memory.add_message(role="AI", content=f"Response: {response.mesg}\n", session_id=session_id)
        return response

    async def upload_file_with_callback(self, file_data: str, callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False, **kwargs) -> None:
        file = base64.b64decode(file_data)
        document_store.upload(file)
        callback("Document Uploaded")

    async def update_prompt_template(self, prompt: str, callback: Callable[[str], None], **kwargs) -> str:
        self.direct_responder._prompt_templates.set(kwargs.get("session_id", DEFAULT_SESSION_ID), prompt)
        callback("Done")

    def update_google_docs_token(self, token: Any) -> Response:
        self._google_docs.set_credentials(token)
        if token:
            return Response(mesg="Logged in")
        else:
            return Response(mesg="Logged out")


class DirectResponse:
    "Designed to answer a question directly"

    prompt = """

Use the following format for your answers:

Human: the input question you must answer
Answer: your answer to the question

Context:
{{$context}}

Chat History:
{{$history}}

Human: {{$input}}
Answer: """

    def __init__(self, kernel: sk.Kernel, character: str = ""):
        self.kernel = kernel
        self.character = character
        self._prompt_templates = PromptTemplate(self.character, self.prompt)

    async def response(self, context: str, history: str, input: str, session_id: Optional[str] = DEFAULT_SESSION_ID, **kwargs) -> Response:
        prompt = self._prompt_templates.get(session_id, self.kernel)
        ctx = self.kernel.create_new_context()
        ctx.variables["context"] = context
        ctx.variables["history"] = history
        ctx.variables["input"] = input
        result = await prompt.invoke_async(context=ctx)
        return Response(mesg=str(result).strip())
