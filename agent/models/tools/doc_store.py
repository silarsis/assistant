import base64
import filetype
import functools
import os
import shutil
import traceback

from collections.abc import Iterable
from pydantic import BaseModel
from pypdf import PdfReader
from typing import Annotated, Optional

from openai import OpenAIError

from langchain_core.documents import BaseDocumentTransformer
# from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Needed for image handling
from models.tools.llm_connect import llm_from_settings
from models.tools.memory import Message

from config import settings

STORE_DIR = './volumes/faiss'
RAW_FILE_DIR = 'raw_files'
METADATA_FILENAME = 'document_metadata.json'


class Document(BaseModel):
    filename: str
    _text: str = None

    class Config:
        arbitrary_types_allowed = True
        frozen = True

    def __init__(self, filename: str):
        super().__init__(filename=filename)

    @property
    async def file_text(self) -> str:
        if not self.filename:
            raise KeyError("No file name was provided. Please provide file data to prompt on.")
        if self._text:
            return self._text
        type_of_file = filetype.guess(self.filename)
        if type_of_file is None:
            raise KeyError("Could not load the file - make sure it has no special characters in the name")
        if type_of_file.MIME.startswith("text"):
            with open(self.filename, "r") as f:
                self._text = f.read()
            return self._text
        if type_of_file.MIME.startswith("image"):
            base64_image = base64.b64encode(self.file_data).decode('utf-8')
            # Find the type of image - is it a flowchart, a gantt chart, or whatever else
            # Then, based on that type, analyze it and return text
            try:
                response = await llm_from_settings().openai(async_client=True).chat.completions.create(
                    model=settings.openai_deployment_name,
                    messages=[Message(role="user", content=[
                        {
                            "type": "text",
                            "text": "Hey there, AI! An image has just been uploaded for analysis. Let's assume the image contains complex features, details, and structures. Could you please perform a deep analysis of this uploaded image and produce a detailed textual description? I want you to capture all the essential elements, colors, patterns, objects, orientations, and interactions within this image. The aim is to create a description so comprehensive that someone could recreate a highly similar image based solely on your textual output. Don't leave out any crucial details that contribute to the overall composition of the image."
                        },
                        {
                            "type": "image_url",
                            "image_url": { "url": f"data:image/jpeg;base64,{base64_image}" }
                        }
                    ])]
                )
            except OpenAIError as e:
                print(f"Error uploading image:\n{traceback.format_exc()}")
                return e.message
            self._text = response.choices[0].message.content
            return self._text
        if type_of_file.MIME == "application/pdf":
            pdf = PdfReader(self.filename)
            self._text = '\n'.join([page.extract_text() for page in pdf.pages])
            return self._text
        raise ValueError("Unsupported file type: {type_of_file}")

    @functools.cached_property
    def file_data(self) -> bytes:
        return open(self.filename, 'rb').read()

    @functools.cached_property
    def saved_filename(self) -> str:
        docname = os.path.basename(self.filename)
        filename = os.path.join(STORE_DIR, RAW_FILE_DIR, docname)
        return filename

    async def save(self) -> None:
        os.makedirs(os.path.join(STORE_DIR, RAW_FILE_DIR), exist_ok=True)
        with open(self.saved_filename + '.txt', 'w') as fd:
            fd.write(await self.file_text)
        if os.path.isfile(self.filename):
            shutil.move(self.filename, self.saved_filename)


class DocStore(BaseModel):
    _embeddings = None
    _text_splitter: BaseDocumentTransformer = None
    _metadata_filename = os.path.join(STORE_DIR, METADATA_FILENAME)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        llmConnector = llm_from_settings()
        self._embeddings = llmConnector.embeddings()
        self._text_splitter = RecursiveCharacterTextSplitter()
        # self._text_splitter = SemanticChunker(self._embeddings)
        # Setup the FAISS repository of docs
        os.makedirs(STORE_DIR, exist_ok=True)

    def _already_loaded(self, docname: str) -> bool:
        " Is this doc already loaded? "
        return docname in self._all_docnames()

    def _all_docnames(self) -> Iterable[str]:
        " List all known docs "
        return ( fn for fn in os.listdir(STORE_DIR) if fn not in (METADATA_FILENAME, RAW_FILE_DIR) )

    def _get_doc_by_docname(self, docname: str) -> FAISS:
        " Return a FAISS Document for the given docid "
        return FAISS.load_local(os.path.join(STORE_DIR, docname), self._embeddings, allow_dangerous_deserialization=True)

    def _all_docs(self) -> Iterable[FAISS]:
        for docname in self._all_docnames():
            yield self._get_doc_by_docname(docname)

    def query(self, query: str) -> list[str]:
        return list(self.search_for_phrases(query))

    async def upload_document(self, doc: Document) -> Annotated[str, "Document ID"]:
        " Takes an entire document as a string, breaks it into elements and saves it in a vectordb "
        base_docname = os.path.basename(doc.filename)
        if base_docname in self._all_docnames():
            return base_docname
        # Really should either hint or figure out the best method for chunking different doc types.
        split_docs = self._text_splitter.split_text(await doc.file_text)
        docs = self._text_splitter.create_documents(split_docs, metadatas=[{'docname': base_docname}] * len(split_docs))
        FAISS.from_documents(docs, self._embeddings).save_local(os.path.join(STORE_DIR, base_docname))
        await doc.save()
        return base_docname

    def delete_document(self, docname: str):
        " Delete the document from the store "
        doc = self._get_doc_by_docname(docname)
        doc.delete(doc.index_to_docstore_id.values()) # This doesn't delete the FAISS object itself, just all the contents
        try:
            shutil.rmtree(os.path.join(STORE_DIR, docname))
        except FileNotFoundError:
            pass # Already not there
        try:
            os.remove(os.path.join(STORE_DIR, RAW_FILE_DIR, docname))
        except FileNotFoundError:
            pass # Already not there
        try:
            os.remove(os.path.join(STORE_DIR, RAW_FILE_DIR, docname + '.txt'))
        except FileNotFoundError:
            pass # Already not there

    def list_documents(self) -> Iterable[Annotated[str, "docname"], Annotated[str, "Full path for file"]]:
        " Return a list of documents "
        for docname in self._all_docnames():
            yield [docname, os.path.join(STORE_DIR, RAW_FILE_DIR, docname)]

    def search_for_phrases(self, query: str, docnames: Optional[list[str]] = None) -> Iterable[str]:
        " Return a list of phrases that are similar to the query, optionally restricted to particular docids "
        for docname in self._all_docnames():
            if (docnames is None) or (docname in docnames):
                for phrase in self._get_doc_by_docname(docname).similarity_search(query):
                    yield phrase.page_content