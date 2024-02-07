import os
import time

from pydantic import BaseModel

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from config import settings

class DocStore(BaseModel):
    _vector_stores: dict = {}
    _chroma_client: chromadb.Client = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
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
            
    def _vector_store(self, collection_name: str = 'LangChainCollection'):
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
            
    def query(self, query: str, collection_name: str = 'LangChainCollection'):
        return self._vector_store(collection_name).query(query)