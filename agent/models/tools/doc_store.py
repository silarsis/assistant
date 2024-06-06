import time

from pydantic import BaseModel

# from langchain.text_splitter import SpacyTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from models.tools.llm_connect import LLMConnect

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from config import settings

class DocStore(BaseModel):
    _vector_stores: dict = {}
    _chroma_client: chromadb.Client = None
    _text_splitter = None
    _embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
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
        if settings.chroma_mode == 'remote':
            self._connect_to_remote_chromadb()
        else:
            self._connect_to_local_chromadb()
        
    def _connect_to_local_chromadb(self):
        self._chroma_client = chromadb.PersistentClient(
            path='./volumes/chroma/local', 
            settings=Settings(anonymized_telemetry=False)
        )

    def _connect_to_remote_chromadb(self):
        while True:
            try:
                self._chroma_client = chromadb.HttpClient(
                    host=settings.chroma_host, 
                    port=settings.chroma_port, 
                    settings=Settings(anonymized_telemetry=False))
            except ValueError as e:
                if 'Could not connect' in str(e):
                    print("Waiting for chromadb...")
                    time.sleep(1)
                else:
                    raise e
            else:
                break
            
    def _cache_key(self, docid: str) -> str:
        return f"doc_{docid}_key"
    
    def _vector_store(self, collection_name: str = 'AssistantCollection'):
        cache_key = self._cache_key(collection_name)
        if cache_key not in self._vector_stores:
            self._vector_stores[cache_key] = self._chroma_client.create_collection(
                embedding_function=self._embeddings,
                name=cache_key, get_or_create=True
            )
        return self._vector_stores[cache_key]
    
    def _already_loaded(self, docid: str) -> bool:
        cache_key = self._cache_key(docid)
        return cache_key in (c.name for c in self._chroma_client.list_collections())
    
    def load_doc(self, docid: str, elements: list[str]):
        if not self._already_loaded(docid):
            self._vector_store(docid).add(documents=elements, ids=[f'{docid}_{i}' for i in range(len(elements))])
            
    def upload(self, doc: str):
        """ Takes an entire document as a string, breaks it into elements and calls load_doc """
        # Really should either hint or figure out the best method for chunking.
        elements = self._text_splitter.create_documents([doc])
        self.load_doc(hash(doc), [e.page_content for e in elements])
            
    def query(self, query: str, collection_name: str = 'AssistantCollection'):
        return self._vector_store(collection_name).query(query)