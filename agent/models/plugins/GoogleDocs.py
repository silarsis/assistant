from typing import Any, Callable, Optional, Annotated

from pydantic import BaseModel

from semantic_kernel.functions.kernel_function_decorator import kernel_function

from googleapiclient.discovery import build
# import to provide google.auth.credentials.Credentials
from google.oauth2.credentials import Credentials
from langchain.docstore.document import Document

import chromadb
from chromadb.config import Settings
# from chromadb.utils import embedding_functions

import re
import time
import os


class GoogleDocLoaderPlugin(BaseModel):
    kernel: Any = None
    _credentials: Credentials = None
    _vector_stores: dict = {}
    _chroma_client: chromadb.Client = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self._embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        # I think I really want to cache the results of these summarise calls
        self._summarize_prompt = self.kernel.create_function_from_prompt(
            function_name="summarize", plugin_name="gdocs", 
            description="Summarize a google document (only works for google docs)",
            prompt="Write a short summary of the following. Do not add any details that are not already there, and if you cannot summarise simply say 'no summary': {{$content}}", 
            max_tokens=2000, temperature=0.2, top_p=0.5)
        self.kernel.register_function_from_method('gdocs', self.load_doc)
        self._connect_to_local_chromadb()
        
    def _connect_to_local_chromadb(self):
        self._chroma_client = chromadb.PersistentClient(path='./volumes/chroma/local', settings=Settings(anonymized_telemetry=False))
    
    def _connect_to_remote_chromadb(self):
        while True:
            try:
                self._chroma_client = chromadb.HttpClient(
                    host=os.environ.get('CHROMA_HOST', 'localhost'), 
                    port=os.environ.get('CHROMA_PORT', '6000'), 
                    settings=Settings(anonymized_telemetry=False))
            except ValueError as e:
                if 'Could not connect' in str(e):
                    print("Waiting for chromadb...")
                    time.sleep(1)
                else:
                    raise e
            else:
                break

    def set_credentials(self, creds: Optional[Credentials] = None) -> str:
        self._credentials = creds
    
    def read_structural_elements(self, elements: list) -> list:
        text = []
        for value in elements:
            if 'paragraph' in value:
                for elem in value.get('paragraph').get('elements'):
                    t = elem.get('textRun', {}).get('content', '')
                    if t:
                        text.append(t)
            elif 'table' in value:
                # The text in table cells are in nested Structural Elements and tables may be
                # nested.
                table = value.get('table')
                for row in table.get('tableRows'):
                    cells = row.get('tableCells')
                    for cell in cells:
                        text.extend(self.read_structural_elements(cell.get('content')))
            elif 'tableOfContents' in value:
                # The text in the TOC is also in a Structural Element.
                toc = value.get('tableOfContents')
                text.extend(self.read_structural_elements(toc.get('content')))
        return text
    
    def _cache_key(self, docid: str) -> str:
        return f"gdoc_{docid}_key"
    
    def _vector_store(self, collection_name: str = 'LangChainCollection'):
        cache_key = self._cache_key(collection_name)
        if cache_key not in self._vector_stores:
            self._vector_stores[cache_key] = self._chroma_client.create_collection(
                # embedding_function=self._embeddings,
                name=cache_key, get_or_create=True
            )
        return self._vector_stores[cache_key]
    
    def _already_loaded(self, docid: str) -> bool:
        cache_key = self._cache_key(docid)
        print(self._chroma_client.list_collections())
        return cache_key in (c.name for c in self._chroma_client.list_collections())
    
    def _fetch_from_gdocs(self, docid: str, creds) -> str:
        service = build("docs", "v1", credentials=creds)
        document = service.documents().get(documentId=docid).execute()
        elements = self.read_structural_elements(document.get('body').get('content'))
        return elements
    
    async def _summarize_elements(self, elements: list, interim: Callable = None) -> str:
        docs = [Document(page_content=t) for t in elements]
        context = self.kernel.create_new_context()
        summaries = []
        
        async def _summarize(block: str) -> str:
            context.variables.set('content', block)
            summarized = await self._summarize_prompt(context=context)
            summary = summarized.result
            if interim:
                interim(summary)
            return summary
            
        block = ''
        for doc in docs:
            block += doc.page_content
            if len(block) > 1000:
                summaries.append(await _summarize(block.strip()))
                block = ''
        if block.strip():
            summaries.append(await _summarize(block.strip()))
        return await _summarize("\n".join(summaries))

    @kernel_function(name="load_doc", description="Load a google document into the vector store")
    async def load_doc(self, docid: Annotated[str, "The google document ID"] = "") -> str:
        " Load a google document into the vector store "
        if not self._credentials:
            return "Unauthorized, please login"
        if docid.startswith('http'):
            # Strip the url
            docid = re.search(r'document/d/([^/]+)/', docid).group(1)
        elements = self._fetch_from_gdocs(docid, self._credentials)
        # Store in the vectordb
        self._vector_store(docid).add(documents=elements, ids=[f'{docid}_{i}' for i in range(len(elements))])
        # Need to store in a separate db for the cleartext, with the same chunking of elements
        # Then, we can use the same ids to retrieve the cleartext for things like summarization
        # Now summarize the doc
        summarized_doc = await self._summarize_elements(elements, interim=None)
        return f"Document loaded successfully. Document Summary: {summarized_doc}"

    
# Thought: Instead of databasing the raw text of the doc, why don't we use gdocs as the database,
# and re-fetch it anytime we need it? Potential for mis-match between vectordb and content for any
# document that's changing, check if there's a "last modified" field in the gdocs api
