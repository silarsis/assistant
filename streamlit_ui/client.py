import json

from websockets.sync.client import connect, ClientConnection

import streamlit as st
from streamlit.connections import ExperimentalBaseConnection

import google_auth_oauthlib

class WSConnection(ExperimentalBaseConnection[ClientConnection]):
    # WSConnection handles the websocket connection to the assistant server.
    # It subclasses ExperimentalBaseConnection to integrate with Streamlit.

    # _connect establishes the websocket connection.

    # recv receives a message from the websocket and returns the payload.

    # send sends a formatted message to the websocket.
    def _connect(self, **kwargs) -> ClientConnection:
        con = connect(st.session_state.uri)
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
            
def google_login(ws_connection):
    # Gets user credentials for Google OAuth and sends them to the websocket
    # connection to authorize access to Google Drive and Docs APIs.
    if "_credentials" not in st.session_state:
        st.session_state._credentials = google_auth_oauthlib.get_user_credentials(
            ["https://www.googleapis.com/auth/drive.readonly", "https://www.googleapis.com/auth/documents.readonly"],
            # app desktop credentials
            "438635256773-rf4rmv51lo436a576enb74t7pc9n8rre.apps.googleusercontent.com",
            "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",
        )
    ws_connection.send(mesg_type="system", command="update_google_docs_token", mesg=st.session_state._credentials.to_json())

    
def receive(ws_connection):
    """Receives a message payload from the websocket connection
    and displays it in the Streamlit UI.

    Args:
        ws_connection: The websocket connection instance.

    Returns:
        None
    """
    payload = ws_connection.recv()
    with st.chat_message("assistant"):
        st.markdown(payload)
        st.session_state.messages.append({"role": "assistant", "content": payload})

def prompt(ws_connection):
    # Prompts user for input, displays input in chat UI,
    # sends input to websocket connection, waits for and displays
    # response from websocket connection
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

def main():
    """Main function to run the Streamlit chatbot UI.

    Sets up the Streamlit UI with title, sidebar inputs, and chat history.
    Creates a websocket connection to send/receive messages to the assistant server.
    Calls prompt() to get user input and display assistant responses in the chat UI.
    """
    st.title("Echo AI Assistant")
    st.sidebar.text_input("LLM API URI", key="uri", value="ws://localhost:10000")
    st.sidebar.text_input("session_id", key="session_id", value="client")
    st.sidebar.checkbox("Hear Thoughts", key="hear_thoughts")

    # Next bits are from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
    # Store and display messages so far
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    ws_connection = WSConnection("OpenAI")
    prompt(ws_connection)


if __name__ == '__main__':
    main()