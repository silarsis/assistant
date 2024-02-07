## Tools
import base64
from typing import List, Optional, Callable, Any, Literal
import os
from dotenv import load_dotenv
import asyncio

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
from models.plugins.CodeGeneration import CodeGenerationPlugin
from semantic_kernel.core_plugins import FileIOPlugin, MathPlugin, TextPlugin, TimePlugin, TextMemoryPlugin

from config import settings

DEFAULT_SESSION_ID = "static"


class Message(BaseModel):
    mesg: str = ""
    type: Literal["request", "response", "thought", "error"] = "request"
    final: bool = False
    
    def __str__(self):
        return self.model_dump_json()
    
    def __init__(self, *args, **kwargs):
        if args:
            kwargs['mesg'] = args[0] # Allow for init without specifying 'mesg='
        super().__init__(**kwargs)
    
class Response(Message):
    type: str = "response"
    final: bool = True
    
class Thought(Message):
    type: str = "thought"

def getKernel(model: Optional[str] = "") -> sk.Kernel:
    load_dotenv(dotenv_path=".env")
    kernel = sk.Kernel()
    model = model or settings.openai_deployment_name or "gpt-4"
    if settings.openai_api_type == "azure":
        print("Azure")
        client = AsyncAzureOpenAI(api_key=settings.openai_api_key, organization=settings.openai_org_id, base_url=settings.openai_api_base)
        kernel.add_chat_service("echo", AzureChatCompletion(settings.openai_deployment_name, async_client=client))
    else:
        print("OpenAI")
        client = AsyncOpenAI(api_key=settings.openai_api_key, organization=settings.openai_org_id, base_url=settings.openai_api_base)
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
        if settings.wolfram_alpha_appid:
            self.guide.import_plugin(WolframAlphaPlugin(wolfram_alpha_appid=settings.wolfram_alpha_appid), "wolfram")
        self.guide.import_plugin(MathPlugin(), "math")
        self.guide.import_plugin(FileIOPlugin(), "fileIO")
        self.guide.import_plugin(TimePlugin(), "time")
        self.guide.import_plugin(TextPlugin(), "text")
        self.guide.import_plugin(TextMemoryPlugin(), "text_memory")
        self.guide.import_plugin(ImageGenerationPlugin(), "image_generation")
        self.guide.import_plugin(ScrapeTextPlugin(), "scrape_text")
        if settings.google_api_key: # Note this relies on the env variable being set, check this
            self.guide.import_plugin(GoogleSearchPlugin(), "google_search")
        self.guide.import_plugin(CodeGenerationPlugin(kernel=self.guide), "code_generation")
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
        print(f"Tools: {[tool.name for tool in tools]}")
        return tools

    async def _plan(self, goal: str, callback: Callable[[str], None], history_context: str, history: str, session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False) -> Message:
        try:
            plan = self.planner.create_plan(goal=goal)
            if hear_thoughts:
                pass
                #callback(Thought(mesg=f"Planning result"))
            result = await plan.invoke()
        except Exception as e:
            print(f"Planning failed: {e}")
            if hear_thoughts:
                await callback(Thought(mesg=str(e)))
            result = ""
        return Thought(mesg=str(result))

    async def rephrase(self, input: str, answer: Message, history: str, history_context: str, session_id: str = DEFAULT_SESSION_ID) -> Message:
        # Rephrase the text to match the character
        # TODO: Is there a way to force calling a particular plugin at the end of all other plugins in the planner?
        # If so, we could force rephrasing that way.
        prompt = PromptTemplate(
            self.default_character, 
            f"""
You were asked "{input}" and your subordinate helpers suggests the answer to be "{answer.mesg}".
Please respond to the user with this answer. If your chat history or context suggests a better answer, please use that instead.
Check the chat history and context for answers to the question also.

Context:
{{$context}}

Chat History:
{{$history}}

Answer: 
            """
        ).get(session_id, self.guide)
        ctx = self.guide.create_new_context()
        ctx.variables['input'] = answer.mesg
        ctx.variables['history'] = history
        ctx.variables['history_context'] = history_context
        result = await prompt.invoke(context=ctx)
        return Thought(mesg=str(result).strip())
    
    async def _pick_best_answer(self, prompt: str, response1: Message, response2: Message) -> str:
        # Ask the guide which is the better response
        prompt_template = f"""
You have been tasked with answering the following: {prompt}.
Select the response that has the most information in it from the following two responses and indicate your selection by sending either 'response 1' or 'response 2':
Response 1: {response1}
Response 2: {response2}
        """
        query = self.guide.create_semantic_function(
            prompt_template=f"You have been tasked with answering the following: {prompt}\nSelect the most informative response from the following two responses and indicate your selection by sending either 'response 1' or 'response 2':\n\nResponse 1:\n{response1}\n\nResponse 2:\n\n{response2}\n", 
            max_tokens=2000, temperature=0.2, top_p=0.5)
        result = await query.invoke()
        print(result)
        print(f"Response 1: {response1}")
        print(f"Response 2: {response2}")
        if '1' in str(result):
            return response1
        return response2
    
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False, **kwargs) -> Message:
        if not prompt:
            await callback(Response(mesg="I'm afraid I don't have enough context to respond. Could you please rephrase your question?", final=True))
            return
        # Convert the prompt to character + history
        history = self.memory.get_formatted_history(session_id=session_id)
        history_context = self.memory.get_context(session_id=session_id)
        self.memory.add_message(role="Human", content=f"Human: {prompt}", session_id=session_id)
        # Now, we go straight into plan, but I wonder if we should check the memory first
        # and the doc storage, and then plan? Hrm. Right now, planning returns a response
        # that doesn't take any chat history or other memory into account, which is a bit
        # frustrating.
        # Temporarily removing the planning, which means removing all the tools :/
        response, direct_response = await asyncio.gather(
            self._plan(prompt, callback, history_context, history, hear_thoughts=hear_thoughts),
            self.direct_responder.response(history_context, history, prompt, session_id=session_id)
        )
        if response:
            response = await self.rephrase(prompt, response, history, history_context, session_id=session_id)
        best_response = await self._pick_best_answer(prompt, response, direct_response)
        self.memory.add_message(role="AI", content=f"Response: {best_response.mesg}\n", session_id=session_id)
        final_response = Response(mesg=best_response.mesg)
        await callback(final_response)

    async def upload_file_with_callback(self, file_data: str, callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False, **kwargs) -> None:
        file = base64.b64decode(file_data)
        document_store.upload(file)
        await callback(Response("Document Uploaded"))

    async def update_prompt_template(self, prompt: str, callback: Callable[[str], None], **kwargs) -> str:
        self.direct_responder._prompt_templates.set(kwargs.get("session_id", DEFAULT_SESSION_ID), prompt)
        await callback("Done")

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

    async def response(self, context: str, history: str, input: str, session_id: Optional[str] = DEFAULT_SESSION_ID, **kwargs) -> Message:
        prompt = self._prompt_templates.get(session_id, self.kernel)
        ctx = self.kernel.create_new_context()
        ctx.variables["context"] = context
        ctx.variables["history"] = history
        ctx.variables["input"] = input
        result = await prompt.invoke(context=ctx)
        return Thought(mesg=str(result).strip())
