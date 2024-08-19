import os
import shutil

from pydantic import BaseModel
from collections.abc import Iterable
from typing import Annotated, Optional

from langchain_core.documents import BaseDocumentTransformer
# from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models.tools.llm_connect import LLMConnect

from config import settings

STORE_DIR = './volumes/faiss'
RAW_FILE_DIR = 'raw_files'
METADATA_FILENAME = 'document_metadata.json'


class DocStore(BaseModel):
    _embeddings = None
    _text_splitter: BaseDocumentTransformer = None
    _doc_store: dict = {}
    _metadata_filename = os.path.join(STORE_DIR, METADATA_FILENAME)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        llmConnector = LLMConnect(
            api_type=settings.openai_api_type,
            api_key=settings.openai_api_key,
            api_base=settings.openai_api_base,
            deployment_name=settings.openai_deployment_name,
            org_id=settings.openai_org_id
        )
        self._embeddings = llmConnector.embeddings()
        self._text_splitter = RecursiveCharacterTextSplitter()
        # self._text_splitter = SemanticChunker(self._embeddings)
        # Setup the FAISS repository of docs
        os.makedirs(STORE_DIR, exist_ok=True)

    def _store_doc(self, docname: str, doc: str, full_docname: str):
        " Store the raw file post transform to text "
        os.makedirs(os.path.join(STORE_DIR, RAW_FILE_DIR), exist_ok=True)
        with open(os.path.join(STORE_DIR, RAW_FILE_DIR, docname + '.txt'), 'w') as fd:
            fd.write(doc)
        if os.path.isfile(full_docname):
            shutil.move(full_docname, os.path.join(STORE_DIR, RAW_FILE_DIR, docname))

    def _already_loaded(self, docname: str) -> bool:
        " Is this doc already loaded? "
        if docname in self._doc_store:
            return True
        return False

    def _all_docnames(self) -> Iterable[str]:
        " List all known docs "
        return ( fn for fn in os.listdir(STORE_DIR) if fn not in (METADATA_FILENAME, RAW_FILE_DIR) )

    def _get_doc_by_docname(self, docname: str) -> FAISS:
        " Return a FAISS Document for the given docid "
        if not self._already_loaded(docname):
            self._doc_store[docname] = FAISS.load_local(os.path.join(STORE_DIR, docname), self._embeddings, allow_dangerous_deserialization=True)
        return self._doc_store[docname]

    def _all_docs(self) -> Iterable[FAISS]:
        for docname in self._all_docnames():
            yield self._get_doc_by_docname(docname)

    def query(self, query: str) -> list[str]:
        return list(self.search_for_phrases(query))

    def upload_document(self, doc: str, docname: str) -> Annotated[str, "Document ID"]:
        " Takes an entire document as a string, breaks it into elements and saves it in a vectordb "
        if self._already_loaded(docname):
            return
        # Really should either hint or figure out the best method for chunking different doc types.
        base_docname = os.path.basename(docname)
        split_docs = self._text_splitter.split_text(doc)
        docs = self._text_splitter.create_documents(split_docs, metadatas=[{'docname': base_docname}] * len(split_docs))
        self._doc_store[docname] = FAISS.from_documents(docs, self._embeddings)
        self._doc_store[docname].save_local(os.path.join(STORE_DIR, base_docname))
        self._store_doc(base_docname, doc, docname)
        return base_docname

    def delete_document(self, docname: str):
        " Delete the document from the store "
        if not self._already_loaded(docname):
            return
        self._doc_store[docname].delete_local()
        del self._doc_store[docname]
        os.remove(os.path.join(STORE_DIR, RAW_FILE_DIR, docname))

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