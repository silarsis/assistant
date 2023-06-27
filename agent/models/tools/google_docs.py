from googleapiclient.discovery import build
# import to provide google.auth.credentials.Credentials
from google.oauth2.credentials import Credentials
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

from pymilvus import utility, Collection, connections

MILVUS_HOST = 'milvus'
MILVUS_PORT = 19530

class GoogleDocLoader:
    def __init__(self, llm=None):
        self._tokens = {}
        self._embeddings = OpenAIEmbeddings(model="ada") # TODO this is maybe wrong for Azure OpenAI
        self._llm = llm
        self._vector_stores = {}
        connections.connect(alias='default', host=MILVUS_HOST, port=MILVUS_PORT)
        
    def set_token(self, token: dict, session_id: str = 'static') -> str: # TODO: This isn't token, this is credentials object
        self._tokens[session_id] = Credentials.from_authorized_user_info(token)
    
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
        return f"gdoc_{docid}"
    
    def _vector_store(self, collection_name: str = 'LangChainCollection') -> Milvus:
        cache_key = self._cache_key(collection_name)
        if cache_key not in self._vector_stores:
            self._vector_stores[cache_key] = Milvus(
                embedding_function=self._embeddings, 
                connection_args={'host':MILVUS_HOST, 'port':MILVUS_PORT},
                collection_name=cache_key
            )
        print(self._vector_stores[cache_key])
        return self._vector_stores[cache_key]
    
    def _already_loaded(self, docid: str) -> bool:
        cache_key = self._cache_key(docid)
        if utility.has_collection(cache_key):
            Collection(cache_key).load()
            return True
        return False
    
    def load_doc(self, docid: str, session_id: str = 'static', interim=None) -> str:
        creds = self._tokens.get(session_id, None)
        if not creds:
            return "No token found"
        if self._already_loaded(docid):
            return "Document already loaded" # TODO can we get the summary saved, or recreate it here?
        service = build("docs", "v1", credentials=creds)
        document = service.documents().get(documentId=docid).execute()
        # Chunk and store the doc in our vector store
        elements = self.read_structural_elements(document.get('body').get('content'))
        self._vector_store(docid).add_texts(elements)
        # Now summarize the doc
        chain = load_summarize_chain(self._llm, chain_type="map_reduce")
        docs = [Document(page_content=t) for t in elements]
        summary = chain.run(input_documents=docs, question="Write a summary within 200 words.")
        return f"Document loaded successfully. Document Summary: {summary}"
    
    def query_doc(self, docid: str, query: str, session_id: str = 'static', interim=None) -> str:
        response = ''
        if not self._already_loaded(docid):
            response = self.load_doc(docid, session_id, interim)
            if interim:
                interim(response)
        # LangChain for querying a doc in Milvus
        return "Not implemented yet, sorry"
        