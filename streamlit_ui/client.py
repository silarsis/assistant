import json
import time

from websockets.sync.client import connect, ClientConnection
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

import streamlit as st
from streamlit.connections import ExperimentalBaseConnection

import google_auth_oauthlib

class WSConnection(ExperimentalBaseConnection[ClientConnection]):
    def _connect(self, **kwargs) -> ClientConnection:
        con = connect(st.session_state.uri)
        return con
    
    def recv(self):
        print("receiving")
        message = self._instance.recv()
        try:
            payload = json.loads(message)['payload']
        except json.decoder.JSONDecodeError:
            print("Garbled message, ignoring...")
        print(f"received {payload}")
        return payload
    
    def send(self, mesg_type: str = 'prompt', mesg: str = '', command: str = ''):
        json_message = {'type':mesg_type, 'prompt':mesg, 'command':'', 'session_id':st.session_state.session_id}
        if st.session_state.hear_thoughts:
            json_message['hear_thoughts'] = True
        self._instance.send(json.dumps(json_message))
            
def google_login(ws_connection):
    if '_credentials' not in st.session_state:
        st.session_state._credentials = google_auth_oauthlib.get_user_credentials(
            ["https://www.googleapis.com/auth/drive.readonly", "https://www.googleapis.com/auth/documents.readonly"], 
            # app desktop credentials
            '438635256773-rf4rmv51lo436a576enb74t7pc9n8rre.apps.googleusercontent.com', 
            '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')
    ws_connection.send(
        mesg_type='system', 
        command='update_google_docs_token', 
        mesg=st.session_state._credentials.to_json())
    
def receive(ws_connection):
    payload = ws_connection.recv()
    with st.chat_message("assistant"):
        st.markdown(payload)
        st.session_state.messages.append({"role": "assistant", "content": payload})
    print("Exiting recv")

def prompt(ws_connection):
    print("Calling prompt")
    if prompt := st.chat_input("Input here"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Send the prompt
        ws_connection.send(mesg=prompt)
        # Wait for a response
        receive(ws_connection)
    print("Exiting prompt")

def main():
    print("Calling main")
    st.title('Echo AI Assistant')
    st.sidebar.text_input('LLM API URI', key='uri', value='ws://localhost:10000')
    st.sidebar.text_input('session_id', key='session_id', value='client')
    st.sidebar.checkbox('Hear Thoughts', key='hear_thoughts')

    # Next bits are from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
    # Store and display messages so far
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    ws_connection = WSConnection('OpenAI')
    prompt(ws_connection)

if __name__ == '__main__':
    main()