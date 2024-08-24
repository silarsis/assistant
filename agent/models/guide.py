## Tools
from collections.abc import Iterable
from typing import Optional, Callable, Any, Literal, Union, Annotated
import asyncio

from pydantic import BaseModel

# from models.tools import apify
from models.tools.doc_store import DocStore, Document
from models.tools.llm_connect import LLMConnect
from models.graph import guide as graph_guide #, upload as upload_guide

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

    async def remember_file(self, filename: bytes, session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False) -> Response:
        try:
            doc = Document(filename=filename)
        except (KeyError, ValueError) as exc:
            return Response(mesg=str(exc))
        document_store = DocStore()
        await document_store.upload_document(doc)
        return Response(mesg="Document Uploaded")

    async def prompt_file_with_callback(self, filename: bytes, callback: Callable[[str], None], session_id: str = DEFAULT_SESSION_ID, hear_thoughts: bool = False) -> GuideMesg:
        try:
            doc = Document(filename=filename)
        except (KeyError, ValueError) as exc:
            await callback(Response(mesg=str(exc)))
            return
        # Commenting out until I can get the flow right
        # final_response = await upload_guide.invoke(
        #     doc.file_text,
        #     callback,
        #     hear_thoughts=hear_thoughts,
        #     session_id=session_id,
        #     character=self.default_character
        # )
        # await callback(Response(final_response))
        await callback(Response(mesg=await doc.file_text))
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