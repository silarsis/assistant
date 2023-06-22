from io import BytesIO
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from langchain.docstore.document import Document

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

class GoogleDocLoader:
    def __init__(self):
        self._tokens = {}
        
    def set_token(self, token: str, session_id: str = 'static') -> str:
        self._tokens[session_id] = token
    
    def load_doc(self, docid: str, session_id: str = 'static') -> str:
        creds = self._tokens.get(session_id, None)
        if not creds:
            return "No token found"

        service = build("drive", "v3", credentials=creds)
        file = service.files().get(fileId=docid, supportsAllDrives=True).execute()
        request = service.files().export_media(fileId=docid, mimeType="text/plain")
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        try:
            while done is False:
                status, done = downloader.next_chunk()
        except HttpError as e:
            if e.resp.status == 404:
                print("File not found: {}".format(docid))
            else:
                print("An error occurred: {}".format(e))
        text = fh.getvalue().decode("utf-8")
        metadata = {
            "source": f"https://docs.google.com/document/d/{docid}/edit",
            "title": f"{file.get('name')}",
        }
        
        return Document(page_content=text, metadata=metadata)
    
    def save_doc(self, doc):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(doc)
        db = FAISS.from_documents(documents, OpenAIEmbeddings()) # TODO: Want to push this to milvus