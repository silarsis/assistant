import os
import json
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
        
    def _store_metadata(self, docid: str, docname: str):
        # import the pickle from the filesystem
        metadata = self._get_metadata()
        metadata[docid] = os.path.basename(docname)
        with open(self._metadata_filename, 'w') as fd:
            json.dump(metadata, fd)
            
    def _get_metadata(self) -> dict:
        try:
            with open(self._metadata_filename, 'r') as fd:
                metadata = json.load(fd)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            metadata = {}
        return metadata
    
    def _store_doc(self, docid: str, doc: str):
        " Store the raw file post transform to text "
        os.makedirs(os.path.join(STORE_DIR, RAW_FILE_DIR), exist_ok=True)
        with open(os.path.join(STORE_DIR, RAW_FILE_DIR, docid), 'w') as fd:
            fd.write(doc)

    def _already_loaded(self, docid: str) -> bool:
        " Is this docid already loaded? "
        if docid in self._doc_store:
            return True
        return False
    
    def _all_docids(self) -> Iterable[str]:
        " List all known docids "
        return ( fn for fn in os.listdir(STORE_DIR) if fn not in (METADATA_FILENAME, RAW_FILE_DIR) )
    
    def _get_doc_by_docid(self, docid: str) -> FAISS:
        " Return a FAISS Document for the given docid "
        if not self._already_loaded(docid):
            self._doc_store[docid] = FAISS.load_local(os.path.join(STORE_DIR, docid), self._embeddings, allow_dangerous_deserialization=True)
        return self._doc_store[docid]

    def _all_docs(self) -> Iterable[FAISS]:
        for docid in self._all_docids():
            yield self._get_doc_by_docid(docid)
            
    def query(self, query: str) -> list[str]:
        return list(self.search_for_phrases(query))

    def upload_document(self, doc: str, docname: str) -> Annotated[str, "Document ID"]:
        " Takes an entire document as a string, breaks it into elements and saves it in a vectordb "
        if docname:
            docid = str(hash(docname))
        else:
            docid = str(hash(doc))
        if self._already_loaded(docid):
            return
        # Really should either hint or figure out the best method for chunking different doc types.
        split_docs = self._text_splitter.split_text(doc)
        docs = self._text_splitter.create_documents(split_docs, metadatas=[{'docname': os.path.basename(docname)}] * len(split_docs))
        self._doc_store[docid] = FAISS.from_documents(docs, self._embeddings)
        self._doc_store[docid].save_local(os.path.join(STORE_DIR, str(docid)))
        self._store_doc(docid, doc)
        self._store_metadata(docid, docname)
        return docid

    def delete_document(self, docid: str):
        " Delete the document from the store "
        if not self._already_loaded(docid):
            return
        self._doc_store[docid].delete_local()
        del self._doc_store[docid]
        os.remove(os.path.join(STORE_DIR, RAW_FILE_DIR, docid))
        metadata = self._get_metadata()
        if docid in metadata:
            del metadata[docid]
            with open(self._metadata_filename, 'w') as fd:
                json.dump(metadata, fd)

    def list_documents(self) -> Iterable[Annotated[str, "docid"], Annotated[str, "docname"], Annotated[str, "Full path for file"]]:
        " Return a list of docids and doc names "
        metadata = self._get_metadata()
        for docid in self._all_docids():
            if docid not in metadata:
                # Remove files that haven't been properly uploaded
                # Remove the directory - possibly not secure, but you're running this as yourself
                shutil.rmtree(os.path.join(STORE_DIR, docid))
                continue
            yield [docid, metadata.get(docid, 'Undefined Filename'), os.path.join(STORE_DIR, RAW_FILE_DIR, docid)]

    def search_for_phrases(self, query: str, docids: Optional[list[str]] = None) -> Iterable[str]:
        " Return a list of phrases that are similar to the query, optionally restricted to particular docids "
        for docid in self._all_docids():
            if (docids is None) or (docid in docids):
                for phrase in self._get_doc_by_docid(docid).similarity_search(query):
                    yield phrase.page_content