## Tools
from collections.abc import Iterable
from typing import Optional, Callable, Any, Literal, Union, Annotated
import asyncio

from pydantic import BaseModel

# Tools for file upload
import filetype
from pypdf import PdfReader

# from models.tools import apify
from models.tools.memory import Memory, Message
from models.tools.openai_uploader import upload_image
from models.tools.doc_store import DocStore
from models.tools.llm_connect import LLMConnect
from models.graph import guide as graph_guide

# New semantic kernel setup
import semantic_kernel as sk

from config import settings

DEFAULT_SESSION_ID = "static"


class GuideMesg(BaseModel):
    mesg: str = ""
    type: Literal["request", "response", "thought", "error"] = "request"
    final: bool = False

    def __str__(self):
        return self.model_dump_json()

    def __init__(self, *args, **kwargs):
        if args:
            kwargs['mesg'] = args[0] # Allow for init without specifying 'mesg='
        super().__init__(**kwargs)

class Response(GuideMesg):
    type: str = "response"
    final: bool = True

class Thought(GuideMesg):
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
        self.default_character = default_character
        self.radioQueue = asyncio.Queue()
        print("Guide initialised")

    async def prompt_file_with_callback(self, filename: bytes, callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False) -> GuideMesg:
        if not filename:
            await callback(Response(mesg="No file name was provided. Please provide file data to prompt on.", final=True))
            return
        # Now want to decide what to do with the file - if it's a document, rip it apart and store it. If it's an image, send it to OpenAI
        type_of_file = filetype.guess(filename)
        if type_of_file is None:
            await callback(Response(mesg="Could not load the file - make sure it has no special characters in the name"))
        if type_of_file.MIME.startswith("text"):
            with open(filename, "r") as f:
                file_data = f.read()
            await self.upload_file_with_callback(file_data, filename, callback, session_id=session_id, hear_thoughts=hear_thoughts)
            await callback(Response(mesg="{type_of_file} File successfully uploaded"))
            return None
        if type_of_file.MIME.startswith("image"):
            response = upload_image(filename)
            final_response = Response(mesg=response)
            await asyncio.gather(
                Memory(session_id).add_message(Message(role="assistant", content=final_response.mesg)),
                callback(final_response)
            )
            return None
        if type_of_file.MIME == "application/pdf":
            pdf = PdfReader(filename)
            file_data = '\n'.join([page.extract_text() for page in pdf.pages])
            await self.upload_file_with_callback(file_data, filename, callback, session_id=session_id, hear_thoughts=hear_thoughts)
            await callback(Response(mesg=f"{type_of_file} File successfully uploaded"))
            return None
        callback(Response(mesg=f"Unsupported file type: {type_of_file}"))
        return None

    async def prompt_with_callback(self, prompt: Union[str,bytes], callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False, **kwargs) -> GuideMesg:
        if not prompt:
            await callback(Response(mesg="I'm afraid I don't have enough context to respond. Could you please rephrase your question?", final=True))
            return

        final_response = await graph_guide.invoke(
            prompt,
            callback,
            hear_thoughts=hear_thoughts,
            session_id=session_id,
            character=self.default_character
        )
        await callback(Response(final_response))
        return None

    async def upload_file_with_callback(self, file_data: str, filename: str, callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False, **kwargs) -> None:
        document_store = DocStore()
        document_store.upload_document(file_data, filename)
        await callback(Response("Document Uploaded", final=True))

    def update_google_docs_token(self, token: Any) -> Response:
        # Won't be working for now
        self._google_docs.set_credentials(token)
        if token:
            return Response(mesg="Logged in")
        else:
            return Response(mesg="Logged out")

    def list_plugins(self) -> dict[str, Any]:
        # Not working for now
        return {}
        return self.guide.plugins

    def documents(self) -> Iterable[Annotated[str, "docid"], Annotated[str, "docname"], Annotated[str, "Full pathname for file"]]:
        return DocStore().list_documents()

    def delete_document(self, docid: str) -> Response:
        return DocStore().delete_document(docid)