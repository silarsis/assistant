import json
import os
import base64
import time
import queue

from websockets.sync.client import connect, ClientConnection

import streamlit as st
from streamlit.connections import BaseConnection

import google_auth_oauthlib

import elevenlabs

from listen import Listen

class WSConnection(BaseConnection[ClientConnection]):
    # WSConnection handles the websocket connection to the assistant server.
    # It subclasses BaseConnection to integrate with Streamlit.

    # _connect establishes the websocket connection.

    # recv receives a message from the websocket and returns the payload.

    # send sends a formatted message to the websocket.
    def _connect(self, **kwargs) -> ClientConnection:
        while True:
            try:
                con = connect(st.session_state.uri)
                st.info("Connected to assistant at " + st.session_state.uri)
                break
            except ConnectionError:
                st.error("Connection error. Retrying...")
                time.sleep(2)
        return con

    def recv(self):
        print("receiving")
        message = self._instance.recv()
        try:
            payload = json.loads(message)["payload"]
        except json.decoder.JSONDecodeError:
            print("Garbled message, ignoring...")
        print(f"received {payload}")
        return payload

    def send(self, mesg_type: str = "prompt", mesg: str = "", command: str = ""):
        json_message = {"type": mesg_type, "prompt": mesg, "command": "", "session_id": st.session_state.session_id}
        if st.session_state.hear_thoughts:
            json_message["hear_thoughts"] = True
        self._instance.send(json.dumps(json_message))
        
class AudioConnection(BaseConnection[Listen]):
    def _connect(self, **kwargs) -> Listen:
        if 'listen_queue' not in st.session_state:
            self.listen_queue = st.session_state.listen_queue = queue.Queue()
        else:
            self.listen_queue = st.session_state.listen_queue
        listener = Listen(self._recv_from_bg_queue)
        if 'listen' in st.session_state and st.session_state.listen:
            listener.start_listening()
        return listener
    
    def _recv_from_bg_queue(self, mesg: str):
        # This is to get the audio message back onto the main thread before processing
        self.listen_queue.put(mesg)
    
    def start_listening(self):
        self._instance.start_listening()
        
    def stop_listening(self):
        self._instance.stop_listening()
        
    def toggle_listening(self):
        if not st.session_state.listen:
            self.stop_listening()
        else:
            self.start_listening()
            
    def speak(self, payload: str = ''):
        if st.session_state.listen:
            self.stop_listening()
        audio_stream = elevenlabs_get_stream(text=payload)
        audio_bytes = b"".join([bytes(a) for a in audio_stream])
        st.markdown(f'<audio autoplay src="data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}" type="audio/mp3"></audio>', unsafe_allow_html=True)
        if st.session_state.listen:
            self.start_listening()
            
    def quit(self):
        self._instance.quit()

def elevenlabs_get_stream(text: str = '') -> bytes:
    elevenlabs.set_api_key(os.environ.get('ELEVENLABS_API_KEY'))
    voices = [os.environ.get('ELEVENLABS_VOICE_1_ID'), os.environ.get('ELEVENLABS_VOICE_2_ID')]
    try:
        audio_stream = elevenlabs.generate(text=text, voice=voices[0], stream=True)
    except elevenlabs.api.error.RateLimitError as err:
        print(str(err), flush=True)
    return audio_stream

def google_login():
    # Gets user credentials for Google OAuth and sends them to the websocket
    # connection to authorize access to Google Drive and Docs APIs.
    if "_credentials" not in st.session_state:
        st.session_state._credentials = google_auth_oauthlib.get_user_credentials(
            ["https://www.googleapis.com/auth/drive.readonly", "https://www.googleapis.com/auth/documents.readonly"],
            # app desktop credentials
            "438635256773-rf4rmv51lo436a576enb74t7pc9n8rre.apps.googleusercontent.com",
            "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",
        )
    st.session_state.ws_connection.send(mesg_type="system", command="update_google_docs_token", mesg=st.session_state._credentials.to_json())

def receive():
    while True:
        payload = st.session_state.ws_connection.recv()
        with st.chat_message("assistant"):
            st.markdown(payload)
            st.session_state.messages.append({"role": "assistant", "content": payload})
            if st.session_state.speak:
                st.session_state.listener.speak(payload)

def process_user_input(prompt: str = ""):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Send the prompt
    st.session_state.ws_connection.send(mesg=prompt)
    
def prompt():
    # Prompts user for input, displays input in chat UI,
    # sends input to websocket connection, waits for and displays
    # response from websocket connection
    if 'listen_queue' in st.session_state:
        try:
            mesg = st.session_state.listen_queue.get_nowait()
            process_user_input(mesg)
        except queue.Empty:
            pass
    if prompt := st.chat_input("Input here"):
        process_user_input(prompt)
    
def main():
    """Main function to run the Streamlit chatbot UI.

    Sets up the Streamlit UI with title, sidebar inputs, and chat history.
    Creates a websocket connection to send/receive messages to the assistant server.
    Calls prompt() to get user input and display assistant responses in the chat UI.
    """
    
    st.title("Echo AI Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    with st.sidebar.expander("Connection Parameters"):
        st.text_input("LLM API URI", key="uri", value="ws://localhost:10000")
        st.text_input("session_id", key="session_id", value="client")
        
    # Connect to our backend agent
    st.session_state.ws_connection = WSConnection("agent")
    # Create the speech and hearing centre
    st.session_state.listener = AudioConnection("speech")
    
    with st.sidebar.expander("Interface Options"):
        st.checkbox("Hear Thoughts", key="hear_thoughts")
        st.checkbox("Speak", key="speak", value=True)
        st.checkbox("Listen (not working yet)", key="listen", value=False, on_change=st.session_state.listener.toggle_listening)
    
    # Next bits are from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
    # Store and display messages so far
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt()
    receive()


if __name__ == '__main__':
    main()