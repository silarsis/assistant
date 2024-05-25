## Tools
import base64
from typing import Optional, Callable, Any, Literal, Union, List
import asyncio
import time
import numpy

from pydantic import BaseModel

# from models.tools import apify
from models.tools.memory import Memory
from models.tools.openai_uploader import upload_image
from models.tools.doc_store import DocStore
from models.tools.llm_connect import LLMConnect

# New semantic kernel setup
import semantic_kernel as sk
from semantic_kernel.planners.sequential_planner import SequentialPlanner
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.exceptions import PlannerException
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.connectors.memory import chroma
from chromadb.config import Settings as chroma_settings
from chromadb.utils import embedding_functions

# from models.plugins.ScrapeText import ScrapeTextPlugin
from models.plugins.WolframAlpha import WolframAlphaPlugin
from models.plugins.GoogleDocs import GoogleDocLoaderPlugin
from models.plugins.GoogleSearch import GoogleSearchPlugin
from models.plugins.ImageGeneration import ImageGenerationPlugin
from models.plugins.ScrapeText import ScrapeTextPlugin
from models.plugins.CrewAI import CrewAIPlugin
from models.plugins.Tools import ToolsPlugin
from semantic_kernel.core_plugins import MathPlugin, TextPlugin, TimePlugin, TextMemoryPlugin

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
    kernel = sk.Kernel()
    if not settings.openai_deployment_name:
        settings.openai_deployment_name = "gpt-4"
    llmConnector = LLMConnect(
        api_type=settings.openai_api_type, 
        api_key=settings.openai_api_key, 
        api_base=settings.openai_api_base, 
        deployment_name=settings.openai_deployment_name, 
        org_id=settings.openai_org_id
    )
    service_id, service = llmConnector.sk()
    kernel.add_service(service)
    return service_id, kernel


class Guide:
    def __init__(self, default_character: str):
        print("Initialising Guide")
        self.service_id, self.guide = getKernel()
        self._setup_planner()
        self.memory = Memory(kernel=self.guide, service_id=self.service_id)
        self.default_character = default_character
        self.direct_responder = DirectResponse(self.guide, character=self.default_character, service_id=self.service_id)
        self.rephrase_responder = RephraseResponse(self.guide, character=self.default_character, service_id=self.service_id)
        self.selector_response = SelectorResponse(self.guide, character=self.default_character, service_id=self.service_id)
        print("Guide initialised")

    def _setup_planner(self):
        print("Setting up the planner and plugins")
        self._google_docs = GoogleDocLoaderPlugin(kernel=self.guide)
        #self.guide.import_plugin_from_object(self._google_docs, "gdocs")
        if settings.wolfram_alpha_appid:
            self.guide.import_plugin_from_object(WolframAlphaPlugin(wolfram_alpha_appid=settings.wolfram_alpha_appid), "wolfram")
        self.guide.import_plugin_from_object(MathPlugin(), "math")
        self.guide.import_plugin_from_object(TimePlugin(), "time")
        self.guide.import_plugin_from_object(TextPlugin(), "text")
        self.guide.import_plugin_from_object(ImageGenerationPlugin(), "image_generation")
        self.guide.import_plugin_from_object(ScrapeTextPlugin(), "scrape_text")
        self.guide.import_plugin_from_object(ToolsPlugin(kernel=self.guide), "tools")
        self.guide.import_plugin_from_object(GoogleSearchPlugin(), "google_search")
        self.radio = self.guide.import_plugin_from_object(CrewAIPlugin(kernel=self.guide), "crew_ai")
        self.radioQueue = asyncio.Queue()
        self.guide.create_function_from_prompt(
            function_name="generate_code", plugin_name="code_generation",
            description="Generage code from a specification",
            prompt="You are an expert developer who has a special interest in secure code.\nGenerate code according to the following specifications:\n{{$input}}", 
            max_tokens=2000, temperature=0.2, top_p=0.5)
        self.guide.import_plugin_from_prompt_directory("agent/prompts", "precanned")
        
        # memory = SemanticTextMemory(storage=chroma.ChromaMemoryStore(), embeddings_generator=self.guide.get_service(self.service_id).client.embeddings) # Didn't work because I forget why
        persist_directory='./volumes/chroma/memory'
        storage = chroma.ChromaMemoryStore(persist_directory=persist_directory, settings=chroma_settings(anonymized_telemetry=False, is_persistent=True, persist_directory=persist_directory))
        embeddings_generator = embedding_functions.DefaultEmbeddingFunction()
        # Monkey patch to make the API line up - check if this is needed or not post 0.9.1b1
        async def call_embeddings_generator(x):
            return numpy.array(embeddings_generator(x))
        embeddings_generator.generate_embeddings = call_embeddings_generator
        if hasattr(storage, '_default_embedding_function'):
            # storage._default_embedding_function = embeddings_generator
            storage._default_embedding_function = None
        # End monkey patching
        memory = SemanticTextMemory(storage=storage, embeddings_generator=embeddings_generator)
        self.guide.import_plugin_from_object(TextMemoryPlugin(memory), "TextMemoryPlugin")
        
        self.planner = SequentialPlanner(self.guide, self.service_id)
        print("Planner created")

    async def _plan(self, goal: str, callback: Callable[[str], None], history_context: str, history: str, session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False) -> Message:
        try:
            plan = await self.planner.create_plan(goal=goal) # Plan
            if hear_thoughts:
                if plan.steps[0].name == 'Not possible to create plan for goal with available functions.\n':
                    thought = plan.steps[0].name
                    return Thought(mesg=thought)
                else:
                    thought = str([(step.name, step.parameters) for step in plan.steps])
                    await callback(Thought(mesg=f"Planning result:\n{thought}\n"))
            result = await plan.invoke(self.guide) # Execute
        except PlannerException as e:
            try:
                if e.args[1].args[0] == 'Not possible to create plan for goal with available functions.\n':
                    return Thought(mesg=e.args[1].args[0])
            except Exception:
                pass
            print(f"Planning failed: {e}")
            if hear_thoughts:
                await callback(Thought(mesg=str(e)))
            result = ""
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
        return await self.rephrase_responder.response(history_context, history, input, answer.mesg)
    
    async def _pick_best_answer(self, prompt: str, response1: Message, response2: Message) -> Message:
        # Ask the guide which is the better response
        result = await self.selector_response.response(prompt=prompt, response1=response1, response2=response2)
        print(result)
        print(f"Response 1: {response1}")
        print(f"Response 2: {response2}")
        if '1' in str(result.mesg):
            return response1
        return response2
    
    async def prompt_file_with_callback(self, filename: bytes, callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False) -> Message:
        if not filename:
            await callback(Response(mesg="No file name was provided. Please provide file data to prompt on.", final=True))
            return
        history = self.memory.get_formatted_history(session_id=session_id)
        history_context = self.memory.get_context(session_id=session_id)
        response = upload_image(filename)
        final_response = await self.rephrase("describe this image", Thought(mesg=response), history, history_context, session_id=session_id)
        await asyncio.gather(
            self.memory.add_message(role="AI", content=f"Response: {final_response.mesg}\n", session_id=session_id),
            callback(final_response)
        )
        return None
    
    async def prompt_with_callback(self, prompt: Union[str,bytes], callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False, **kwargs) -> Message:
        if not prompt:
            await callback(Response(mesg="I'm afraid I don't have enough context to respond. Could you please rephrase your question?", final=True))
            return
        # Convert the prompt to character + history
        history = self.memory.get_formatted_history(session_id=session_id)
        history_context = self.memory.get_context(session_id=session_id)
        _, response, direct_response = await asyncio.gather(
            self.memory.add_message(role="Human", content=prompt, session_id=session_id),
            self._plan(prompt, callback, history_context, history, hear_thoughts=hear_thoughts),
            self.direct_responder.response(history_context, history, prompt, session_id=session_id)
        )
        if response:
            response = await self.rephrase(prompt, response, history, history_context, session_id=session_id)
        best_response = await self._pick_best_answer(prompt, response, direct_response)
        final_response = Response(mesg=best_response.mesg)
        await asyncio.gather(
            self.memory.add_message(role="AI", content=best_response.mesg, session_id=session_id),
            callback(final_response)
        )
        return None
    
    async def generate_crew(self, goal: str) -> List[List[str]]:
        # Call the CrewAI plugin to generate a crew
        gen_crew = self.guide.func("precanned", "generate_crew")
        result = await self.guide.invoke(gen_crew, goal=goal)
        return result

    async def upload_file_with_callback(self, file_data: str, callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False, **kwargs) -> None:
        file = base64.b64decode(file_data)
        document_store = DocStore()
        document_store.upload(file)
        await callback(Response("Document Uploaded"))

    async def update_prompt_template(self, prompt: str, callback: Callable[[str], None], **kwargs) -> str:
        # self.direct_responder._prompt_templates.set(kwargs.get("session_id", DEFAULT_SESSION_ID), prompt)
        await callback("Not Done")

    def update_google_docs_token(self, token: Any) -> Response:
        self._google_docs.set_credentials(token)
        if token:
            return Response(mesg="Logged in")
        else:
            return Response(mesg="Logged out")
        
    def list_plugins(self) -> List[Any]:
        return self.guide.plugins


class DirectResponse:
    "Designed to answer a question directly"

    prompt = """
Time: {{$time}}

Use the following format for your answers:

Human: the input question you must answer
Answer: your answer to the question

Context:
{{$context}}

Chat History:
{{$history}}

Human: {{$input}}
Answer: """

    def __init__(self, kernel: sk.Kernel, service_id: str = '', character: str = ""):
        self.character = character # So we can support changing it later
        self.kernel = kernel
        req_settings = kernel.get_service(service_id).get_prompt_execution_settings_class()(service_id=service_id)
        req_settings.max_tokens = 2000
        req_settings.temperature = 0.2
        req_settings.top_p = 0.5
        self.prompt_template_config = sk.PromptTemplateConfig(
            template=character + self.prompt, 
            name="direct_response", 
            input_variables=[
                InputVariable(name="time", description="The current time", required=True),
                InputVariable(name="context", description="The context of the conversation", required=True),
                InputVariable(name="history", description="The chat history", required=True),
                InputVariable(name="input", description="The input question you must answer", required=True),
            ],
            execution_settings=req_settings
        )
        self.chat_fn = self.kernel.create_function_from_prompt(
            function_name="direct_response", plugin_name="direct_response",
            description="Directly answer a question",
            prompt_template_config=self.prompt_template_config
        )

    async def response(self, context: str, history: str, input: str, **kwargs) -> Message:
        try:
            result = await self.kernel.invoke(self.chat_fn, context=context, history=history, input=input, time=time.asctime())
        except sk.exceptions.kernel_exceptions.KernelInvokeException as e:
            print(f"Direct response failed: {e}")
            return Thought(mesg=str(e))
        return Thought(mesg=str(result).strip())

class RephraseResponse:
    "Designed to rephrase an answer to match the character's style"

    prompt = """
Time: {{$time}}

You were asked "{{$input}}" and your subordinate helpers suggests the answer to be "{{$answer_mesg}}".
Please respond to the user with this answer. If your chat history or context suggests a better answer, please use that instead.
Check the chat history and context for answers to the question also.

Context:
{{$context}}

Chat History:
{{$history}}

Human: {{$input}}
Answer:  """

    def __init__(self, kernel: sk.Kernel, service_id: str = '', character: str = ""):
        self.character = character # So we can support changing it later
        self.kernel = kernel
        req_settings = kernel.get_service(service_id).get_prompt_execution_settings_class()(service_id=service_id)
        req_settings.max_tokens = 2000
        req_settings.temperature = 0.2
        req_settings.top_p = 0.5
        self.prompt_template_config = sk.PromptTemplateConfig(
            template=character + self.prompt, 
            name="rephrase_response", 
            input_variables=[
                InputVariable(name="time", description="The current time", required=True),
                InputVariable(name="context", description="The context of the conversation", required=True),
                InputVariable(name="history", description="The chat history", required=True),
                InputVariable(name="input", description="The input question you must answer", required=True),
                InputVariable(name="answer_mesg", description="The answer to rephrase", required=True)
            ],
            execution_settings=req_settings
        )
        self.chat_fn = self.kernel.create_function_from_prompt(
            function_name="rephrase_response", plugin_name="rephrase_response",
            description="Rephrase an answer to match the character's style",
            prompt_template_config=self.prompt_template_config
        )

    async def response(self, context: str, history: str, input: str, answer_mesg: str, **kwargs) -> Message:
        try:
            result = await self.kernel.invoke(self.chat_fn, context=context, history=history, input=input, answer_mesg=answer_mesg, time=time.asctime())
        except sk.exceptions.kernel_exceptions.KernelInvokeException as e:
            print(f"Rephrase response failed: {e}")
            return Thought(mesg=str(e)) # Should we just return the original here?
        return Thought(mesg=str(result).strip())

class SelectorResponse:
    "Designed to select between responses"

    prompt = """

You have been tasked with answering the following: {{$prompt}}.
Select the most informative response from the following two responses and indicate your selection by sending either 'response 1' or 'response 2'.
If the responses are factually different, trust the first one more. If the first one is not informative, trust the second one more.

Response 1: {{$response1}}
Response 2: {{$response2}}

Answer: """

    def __init__(self, kernel: sk.Kernel, service_id: str = '', character: str = ""):
        self.character = character # So we can support changing it later
        self.kernel = kernel
        req_settings = kernel.get_service(service_id).get_prompt_execution_settings_class()(service_id=service_id)
        req_settings.max_tokens = 2000
        req_settings.temperature = 0.2
        req_settings.top_p = 0.5
        self.prompt_template_config = sk.PromptTemplateConfig(
            template=character + self.prompt, 
            name="rephrase_response", 
            input_variables=[
                InputVariable(name="prompt", description="The original question", required=True),
                InputVariable(name="response1", description="The first response", required=True),
                InputVariable(name="response2", description="The second response", required=True)
            ],
            execution_settings=req_settings
        )
        self.chat_fn = self.kernel.create_function_from_prompt(
            function_name="selector_response", plugin_name="selector_response",
            description="Select between responses",
            prompt_template_config=self.prompt_template_config
        )

    async def response(self, prompt: str, response1: str, response2: str, **kwargs) -> Message:
        try:
            result = await self.kernel.invoke(self.chat_fn, prompt=prompt, response1=response1, response2=response2)
        except sk.exceptions.kernel_exceptions.KernelInvokeException as e:
            print(f"Selector response failed: {e}")
            return Thought(mesg=str(e)) # Should we just return the first one here?
        return Thought(mesg=str(result).strip())