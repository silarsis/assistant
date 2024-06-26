## Tools
from typing import Optional, Callable, Any, Literal, Union, List
import asyncio
import time

from pydantic import BaseModel

# Tools for file upload
import magic
from pypdf import PdfReader

import dspy

# from models.tools import apify
from models.tools.memory import Memory
from models.tools.openai_uploader import upload_image
from models.tools.doc_store import DocStore
from models.tools.llm_connect import LLMConnect

# New semantic kernel setup
import semantic_kernel as sk
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from semantic_kernel.planners.sequential_planner import SequentialPlanner
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.exceptions import PlannerException
from semantic_kernel.exceptions.kernel_exceptions import KernelInvokeException

# from models.plugins.ScrapeText import ScrapeTextPlugin
from models.plugins.WolframAlpha import WolframAlphaPlugin
from models.plugins.GoogleDocs import GoogleDocLoaderPlugin
from models.plugins.GoogleSearch import GoogleSearchPlugin
from models.plugins.ImageGeneration import ImageGenerationPlugin
from models.plugins.ScrapeText import ScrapeTextPlugin
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
    _ = llmConnector.dspy() # Doing this to ensure it's initialised before needed.
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
        #self.guide.add_plugin(self._google_docs, "gdocs")
        if settings.wolfram_alpha_appid:
            self.guide.add_plugin(WolframAlphaPlugin(wolfram_alpha_appid=settings.wolfram_alpha_appid), "wolfram")
        self.guide.add_plugin(MathPlugin(), "math")
        self.guide.add_plugin(TimePlugin(), "time")
        self.guide.add_plugin(TextPlugin(), "text")
        self.guide.add_plugin(ImageGenerationPlugin(), "image_generation")
        self.guide.add_plugin(ScrapeTextPlugin(), "scrape_text")
        self.guide.add_plugin(ToolsPlugin(kernel=self.guide), "tools")
        self.guide.add_plugin(GoogleSearchPlugin(), "google_search")
        self.radioQueue = asyncio.Queue()
        self.guide.add_function(
            function_name="generate_code", plugin_name="code_generation",
            description="Generage code from a specification",
            prompt="You are an expert developer who has a special interest in secure code.\nGenerate code according to the following specifications:\n{{$input}}", 
            max_tokens=2000, temperature=0.2, top_p=0.5)
        self.guide.add_plugin(parent_directory="prompts", plugin_name="precanned")
        
        self.planner = SequentialPlanner(self.guide, self.service_id)
        print("Planner created")

    async def _plan(self, goal: str, callback: Callable[[str], None], history_context: str, history: str, session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False) -> Message:
        try:
            plan = await self.planner.create_plan(goal)
            if hear_thoughts:
                thought = str([(step.name, step.parameters) for step in plan.steps])
                await callback(Thought(mesg=f"Planning result:\n{thought}\n"))
            result = await plan.invoke(kernel=self.guide)
        except PlannerException as e:
            try:
                if e.args[0] == 'Not possible to create plan for goal with available functions.\n':
                    return Thought(mesg=e.args[0])
            except Exception:
                pass
            print(f"Planning failed with PlannerException: {e}")
            if hear_thoughts:
                await callback(Thought(mesg=str(e)))
            result = f"PlannerException: {e.args[0]}"
        except Exception as e:
            try:
                if 'APITimeoutError' in e.args[0]:
                    return Thought(mesg="Request timed out - check the network connection to your LLM")
            except Exception:
                pass
            print(f"Planning failed: {e}")
            if hear_thoughts:
                await callback(Thought(mesg=str(e)))
            result = f"Exception: {e.args[0]}"
        return Thought(mesg=str(result))

    async def rephrase(self, input: str, answer: Message, history: str, history_context: str, session_id: str = DEFAULT_SESSION_ID) -> Message:
        # Rephrase the text to match the character
        # TODO: Is there a way to force calling a particular plugin at the end of all other plugins in the planner?
        # If so, we could force rephrasing that way.
        return await self.rephrase_responder.response(history_context, history, input, answer.mesg)
    
    async def _pick_best_answer(self, prompt: str, responses: List[Message]) -> Message:
        # Ask the guide which is the better response
        #return Response(mesg="# Planner:\n\n" + response1.mesg + "\n# Direct:\n\n" + response2.mesg) # Temporarily hardcoding to get the long-form response
        result = await self.selector_response.response(prompt=prompt, responses=responses)
        print(f"{result.mesg} was selected as the best response")
        print(f"Responses: {responses}")
        if '1' in str(result.mesg):
            return responses[0]
        if '2' in str(result.mesg):
            return responses[1]
        if '3' in str(result.mesg):
            return responses[2]
        return responses[1]
    
    async def prompt_file_with_callback(self, filename: bytes, callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False) -> Message:
        if not filename:
            await callback(Response(mesg="No file name was provided. Please provide file data to prompt on.", final=True))
            return
        history = self.memory.get_formatted_history(session_id=session_id)
        history_context = self.memory.get_context(session_id=session_id)
        # Now want to decide what to do with the file - if it's a document, rip it apart and store it. If it's an image, send it to OpenAI
        filetype = magic.from_file(filename, mime=True)
        if filetype.startswith("text"):
            with open(filename, "r") as f:
                file_data = f.read()
            # file_data = base64.b64decode(file_data)
            await self.upload_file_with_callback(file_data, callback, session_id=session_id, hear_thoughts=hear_thoughts)
            await callback(Thought(mesg="File successfully uploaded", final=True))
            return None
        if filetype.startswith("image"):
            response = upload_image(filename)
            final_response = await self.rephrase("describe this image", Thought(mesg=response), history, history_context, session_id=session_id)
            final_response.final = True
            await asyncio.gather(
                self.memory.add_message(role="AI", content=f"Response: {final_response.mesg}\n", session_id=session_id),
                callback(final_response)
            )
            return None
        if filetype.startswith("application/pdf"):
            pdf = PdfReader(filename)
            file_data = '\n'.join([page.extract_text() for page in pdf.pages])
            await self.upload_file_with_callback(file_data, callback, session_id=session_id, hear_thoughts=hear_thoughts)
            await callback(Thought(mesg=f"{filetype} File successfully uploaded", final=True))
            return None
        callback(Response(mesg=f"Unsupported file type: {filetype}"))
        return None
    
    async def run_dspy(self, prompt: str):
        generate_answer = DSPYResponse()
        try:
            pred = generate_answer(question=prompt)
        except Exception as e:
            print(f"Failed to run DSPY: {e}")
            return Thought(mesg=str(e))
        return Thought(str(pred))
    
    async def prompt_with_callback(self, prompt: Union[str,bytes], callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False, **kwargs) -> Message:
        if not prompt:
            await callback(Response(mesg="I'm afraid I don't have enough context to respond. Could you please rephrase your question?", final=True))
            return
        # Convert the prompt to character + history
        history = self.memory.get_formatted_history(session_id=session_id)
        history_context = self.memory.get_context(session_id=session_id)
        
        async def rephrase(response: Message) -> Message:
            return await self.rephrase(prompt, await response, history, history_context, session_id=session_id)
        
        tasks = [asyncio.create_task(coro) for coro in [
            self.memory.add_message(role="Human", content=prompt, session_id=session_id),
            rephrase(self._plan(prompt, callback, history_context, history, hear_thoughts=hear_thoughts)),
            rephrase(self.direct_responder.response(history_context, history, prompt, session_id=session_id)),
            rephrase(self.run_dspy(prompt))
        ]]
        await asyncio.wait(tasks, timeout=10)
        results = [ task.result() if task.done() else "Failed to complete in time" for task in tasks ]
        best_response = await self._pick_best_answer(prompt, results[1:])
        final_response = Response(mesg=best_response.mesg)
        await asyncio.gather(
            self.memory.add_message(role="AI", content=best_response.mesg, session_id=session_id),
            callback(final_response)
        )
        return None
    
    async def generate_crew(self, goal: str) -> List[List[str]]:
        # Call the CrewAI plugin to generate a crew - no longer used, deprecating crewai but might replace with something of my own
        gen_crew = self.guide.func("precanned", "generate_crew")
        result = await self.guide.invoke(gen_crew, goal=goal)
        return result

    async def upload_file_with_callback(self, file_data: str, callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False, **kwargs) -> None:
        document_store = DocStore()
        document_store.upload(file_data)
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
        self.prompt_template_config = PromptTemplateConfig(
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
        self.chat_fn = self.kernel.add_function(
            function_name="direct_response", plugin_name="direct_response",
            description="Directly answer a question",
            prompt_template_config=self.prompt_template_config
        )

    async def response(self, context: str, history: str, input: str, **kwargs) -> Message:
        try:
            result = await self.kernel.invoke(self.chat_fn, context=context, history=history, input=input, time=time.asctime())
        except KernelInvokeException as exc:
            return Thought(mesg=str(exc) + " - Possible timeout?")
        except Exception as e:
            try:
                if 'APITimeoutError' in e.args[0]:
                    return Thought(mesg="API Timed out - check the network connection to your LLM")
            except Exception:
                result = str(e)
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
        self.prompt_template_config = PromptTemplateConfig(
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
        self.chat_fn = self.kernel.add_function(
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
Select the most informative response from the following responses and indicate your selection by sending either 'response 1' or 'response 2' or 'response 3'.

Response 1: {{$response1}}
Response 2: {{$response2}}
Response 3: {{$response3}}

Answer: """

    def __init__(self, kernel: sk.Kernel, service_id: str = '', character: str = ""):
        self.character = character # So we can support changing it later
        self.kernel = kernel
        req_settings = kernel.get_service(service_id).get_prompt_execution_settings_class()(service_id=service_id)
        req_settings.max_tokens = 2000
        req_settings.temperature = 0.2
        req_settings.top_p = 0.5
        self.prompt_template_config = PromptTemplateConfig(
            template=character + self.prompt, 
            name="rephrase_response", 
            input_variables=[
                InputVariable(name="prompt", description="The original question", required=True),
                InputVariable(name="response1", description="The first response", required=True),
                InputVariable(name="response2", description="The second response", required=True),
                InputVariable(name="response3", description="The third response", required=True),
            ],
            execution_settings=req_settings
        )
        self.chat_fn = self.kernel.add_function(
            function_name="selector_response", plugin_name="selector_response",
            description="Select between responses",
            prompt_template_config=self.prompt_template_config
        )

    async def response(self, prompt: str, responses: List[Message], **kwargs) -> Message:
        kwargs = { # XXX Figure out how to do a list here
            "prompt": prompt,
            "response1": isinstance(responses[0], Exception) and str(responses[0]) or responses[0].mesg,
            "response2": isinstance(responses[1], Exception) and str(responses[1]) or responses[1].mesg,
            "response3": isinstance(responses[2], Exception) and str(responses[2]) or responses[2].mesg
        }
        try:
            result = await self.kernel.invoke(self.chat_fn, **kwargs)
        except KernelInvokeException as e:
            print(f"Selector response failed: {e}")
            return Thought(mesg=str(e)) # Should we just return the first one here?
        return Thought(mesg=str(result).strip())
    
class DSPYResponse(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought('question -> answer')
        
    def forward(self, question: str):
        pred = self.predict(question=question)
        return dspy.Prediction(answer=pred.answer)
