from pydantic import BaseModel
from typing import Annotated

# from langchain.text_splitter import SpacyTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from models.tools.llm_connect import LLMConnect

import pymilvus
from milvus import default_server

from config import settings

class DocStore(BaseModel):
    _vector_stores: dict = {}
    _docstore_client: pymilvus.MilvusClient = None
    _text_splitter = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        llmConnector = LLMConnect(
            api_type=settings.openai_api_type, 
            api_key=settings.openai_api_key, 
            api_base=settings.openai_api_base, 
            deployment_name=settings.openai_deployment_name, 
            org_id=settings.openai_org_id
        )
        self._text_splitter = SemanticChunker(llmConnector.embeddings())
        self._connect()
        
    def _connect(self):
        if settings.docstore_mode == 'remote':
            self._connect_to_remote_docstore()
        else:
            self._connect_to_local_docstore()
        
    def _connect_to_remote_docstore(self):
        self._docstore_client = pymilvus.connections.connect(
            alias="default",
            address=settings.milvus_address,
            secure=True,
            user=settings.milvus_username,
            password=settings.milvus_api_token,
        )
            
    def _connect_to_local_docstore(self):
        if default_server.running:
            return
        default_server.set_base_dir("./volumes/milvus")
        default_server.start()
        settings.milvus_address = f'127.0.0.1:{default_server.listen_port}'
        self._docstore_client = pymilvus.connections.connect(host='127.0.0.1', port=default_server.listen_port)
            
    def _cache_key(self, docid: str) -> str:
        return f"doc_{docid}_key"
    
    def _vector_store(self, collection_name: str = 'AssistantCollection'):
        cache_key = self._cache_key(collection_name)
        if cache_key not in self._vector_stores:
            self._vector_stores[cache_key] = pymilvus.Collection(name=f"{settings.docstore_collection_prefix}{collection_name}", using='default')
        return self._vector_stores[cache_key]
    
    def _already_loaded(self, docid: Annotated[str, "The document ID"]) -> bool:
        if self._docstore_client is None:
            self._connect()
        cache_key = self._cache_key(docid)
        return cache_key in (c.name for c in self._docstore_client.list_collections())
    
    def load_doc(self, docid: Annotated[str, "The document ID"], elements: list[str]):
        if not self._already_loaded(docid):
            self._vector_store(docid).insert(collection_name=docid, data=elements)
            self._vector_store(docid).add(documents=elements, ids=[f'{docid}_{i}' for i in range(len(elements))])
            
    def upload(self, doc: str):
        """ Takes an entire document as a string, breaks it into elements and calls load_doc """
        # Really should either hint or figure out the best method for chunking.
        elements = self._text_splitter.create_documents([doc])
        self.load_doc(hash(doc), [e.page_content for e in elements])
            
    def query(self, query: str, collection_name: str = 'AssistantCollection'):
        return self._vector_store(collection_name).query(query)