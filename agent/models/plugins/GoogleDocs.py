from typing import Any, Callable, Optional, Annotated

from pydantic import BaseModel

from semantic_kernel.functions.kernel_function_decorator import kernel_function

from googleapiclient.discovery import build
# import to provide google.auth.credentials.Credentials
from google.oauth2.credentials import Credentials
from langchain.docstore.document import Document

from models.tools.doc_store import DocStore

import re


class GoogleDocLoaderPlugin(BaseModel):
    kernel: Any = None
    _credentials: Credentials = None
    _vector_stores: dict = {}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._doc_store = DocStore()
        # I think I really want to cache the results of these summarise calls
        self._summarize_prompt = self.kernel.add_function(
            plugin_name="gdocs",
            function_name="summarize_gdoc", 
            description="Summarize a document",
            prompt="Write a short summary of the following. Do not add any details that are not already there, and if you cannot summarise simply say 'no summary': {{$content}}", 
            max_tokens=2000, temperature=0.2, top_p=0.5)
        self.kernel.add_function(plugin_name='gdocs', function=self.load_gdoc)
        self.kernel.add_function(plugin_name='gdocs', function=self.scrape_gdoc)

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
    
    def _fetch_from_gdocs(self, docid: str, creds) -> str:
        service = build("docs", "v1", credentials=creds)
        document = service.documents().get(documentId=docid).execute()
        elements = self.read_structural_elements(document.get('body').get('content'))
        return elements
    
    async def _summarize_elements(self, elements: list, interim: Callable = None) -> str:
        docs = [Document(page_content=t) for t in elements]
        summaries = []
        
        async def _summarize(block: str) -> str:
            summarized = await self.kernel.invoke(self._summarize_prompt, content=block)
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

    @kernel_function(name="load_gdoc", description="Load a google document into the vector store, ready for future reference. Only use this if you want to commit a document to memory, not if you want to work on the content directly")
    async def load_gdoc(self, docid: Annotated[str, "The google document ID"] = "") -> str:
        " Load a google document into the vector store "
        if not self._credentials:
            return "Unauthorized, please login"
        if docid.startswith('http'):
            # Strip the url
            docid = re.search(r'document/d/([^/]+)/', docid).group(1)
        elements = self._fetch_from_gdocs(docid, self._credentials)
        # Store in the vectordb
        self._doc_store.load_doc(docid, elements)
        summarized_doc = await self._summarize_elements(elements, interim=None)
        return f"Document loaded successfully. Document Summary: {summarized_doc}"
    
    @kernel_function(name='scrape_gdoc', description='Input: URL or docid for a google doc. Output: Scraped text of the document in plain text.')
    async def scrape_gdoc(self, docid: Annotated[str, "The google document ID"] = "") -> str:
        " Scrape a google document for text and return all the content "
        if not self._credentials:
            return "Unauthorized, please login"
        if docid.startswith('http'):
            # Strip the url
            docid = re.search(r'document/d/([^/]+)/', docid).group(1)
        elements = self._fetch_from_gdocs(docid, self._credentials)
        return ''.join(elements)
        
    
# Thought: Instead of databasing the raw text of the doc, why don't we use gdocs as the database,
# and re-fetch it anytime we need it? Potential for mis-match between vectordb and content for any
# document that's changing, check if there's a "last modified" field in the gdocs api
