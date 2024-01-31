import os
import base64
import queue
import asyncio
import requests
import time

from urllib.parse import quote

import streamlit as st
from streamlit.connections import BaseConnection

import google_auth_oauthlib

import pyaudio
import wave
import elevenlabs

from listen import Listen
from prompt_parser import Parser


TTS_HOST=os.environ.get("TTS_HOST", "localhost")
TTS_PORT=os.environ.get("TTS_PORT", "5002")
        
class AudioConnection(BaseConnection[Listen]):
    def _connect(self, **kwargs) -> Listen:
        if 'listener' not in st.session_state:
            self.listen_queue = st.session_state.listen_queue = queue.Queue()
            listener = Listen(self._recv_from_bg_queue)
            if 'listen' in st.session_state and st.session_state.listen:
                listener.start_listening()
        else:
            self.listen_queue = st.session_state.listen_queue
            listener = st.session_state.listener
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
        if os.environ.get('ELEVENLABS_API_KEY'):
            audio_stream = elevenlabs_get_stream(text=payload)
            audio_bytes = b"".join([bytes(a) for a in audio_stream])
            st.markdown(f'<audio autoplay src="data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}" type="audio/mp3"></audio>', 
                unsafe_allow_html=True)
        else:
            # TTS here
            try:
                audio_stream = tts_get_stream(payload)
                # st.markdown(f'audio autoplay src="data:audio/wav;base64,{base64.b64encode(audio_stream).decode()}" type="audio/wave"></audio>', 
                #     unsafe_allow_html=True)
            except requests.exceptions.ConnectionError as e:
                print(f"Speech error, not speaking: {e}")
        if st.session_state.listen:
            self.start_listening()
            
    def quit(self):
        self._instance.quit()
        
class AgentConnector(BaseConnection[Parser]):
    def _connect(self, **kwargs) -> Parser:
        self.bot = Parser()
        return self.bot
    
    async def send(self, prompt, **kwargs):
        # Called in the main thread, puts sent messages on the queue for the background thread to consume
        response = await self.bot.prompt_with_callback(prompt, callback=self.recv_from_bot)
        self.recv_from_bot(response)
        
    def recv_from_bot(self, response):
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            if st.session_state.speak:
                st.session_state.listener.speak(response)
        
def tts_get_stream(text: str) -> bytes:
    q_text = quote(text)
    print("Requesting TTS from Local TTS instance")
    with requests.get(f"http://{TTS_HOST}:{TTS_PORT}/api/tts?text={q_text}&speaker_id=p364&style_wav=&language_id=", stream=True) as wav:
        p = pyaudio.PyAudio() # Need to remove this and send it back properly
        wf = wave.Wave_read(wav.raw)
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=1,
                        rate=22050,
                        output=True)
        while len(data := wf.readframes(1024)):
            stream.write(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
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
        result = st.session_state.agent.bot.update_google_docs_token(st.session_state._credentials)
        with st.chat_message("assistant"):
            st.markdown(result)
            
def google_logout():
    if "_credentials" in st.session_state:
        st.session_state._credentials = None
        result = st.session_state.agent.bot.update_google_docs_token(st.session_state._credentials)
        with st.chat_message("assistant"):
            st.markdown(result)

async def process_user_input(prompt):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Send the prompt
    await st.session_state.agent.send(prompt,
        hear_thoughts=st.session_state.hear_thoughts, 
        session_id=st.session_state.session_id)
    
def process_file_upload():
    file = st.session_state.file_uploader
    print("Got an uploaded file")
    # Upload the file to the upstream agent
    
async def check_for_audio():
    if 'listen_queue' in st.session_state:
        try:
            mesg = st.session_state.listen_queue.get_nowait()
            await process_user_input(mesg)
        except queue.Empty:
            pass
        
def main():
    """Main function to run the Streamlit chatbot UI.

    Sets up the Streamlit UI with title, sidebar inputs, and chat history.
    Creates a websocket connection to send/receive messages to the assistant server.
    Calls prompt() to get user input and display assistant responses in the chat UI.
    """
    
    st.title("Echo AI Assistant")
    st.session_state.session_id = 'streamlit'
    
    # Turn the status bar into a context object that only renders if it's needed?
    status = st.sidebar.status("Setting up backend", state="running", expanded=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Connect to our backend agent
    if True: # "agent" not in st.session_state:
        status.write("Creating agent...")
        try:
            st.session_state.agent = AgentConnector("agent")
        except Exception as e:
            status.write(str(e))
            status.update(label="Failed creating agent", state="error", expanded=True)
            st.stop()
        status.write("Done creating agent")
    else:
        status.write("Using existing agent")
        
    # Create the speech and hearing centre
    if "listener" not in st.session_state:
        status.write("Creating speech and hearing...")
        try:
            st.session_state.listener = AudioConnection("speech")
        except Exception as e:
            status.write(str(e))
            status.update(label="Failed creating speech and hearing...", state="error", expanded=True)
            st.stop()
        status.write("Done creating speech and hearing")
    else:
        status.write("Using existing speech and hearing")
    
    with st.sidebar.expander("Interface Options"):
        st.checkbox("Hear Thoughts", key="hear_thoughts")
        st.checkbox("Speak", key="speak", value=True)
        st.checkbox("Listen", key="listen", value=False, on_change=st.session_state.listener.toggle_listening)
    
    if "_credentials" in st.session_state:
        st.sidebar.button("Logout from Google", key="google_login_button", on_click=google_logout)
    else:
        st.sidebar.button("Login to Google", key="google_login_button", on_click=google_login)
    
    # Next bits are from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
    # Store and display messages so far
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.sidebar.file_uploader("Upload Files", key="file_uploader", accept_multiple_files=True, on_change=process_file_upload)
    status.update(label="Setup Complete", state="complete", expanded=False)

async def process_input_output():
    print("Starting IO")
    if prompt := st.chat_input("Input here", key="chat_input"):
        await process_user_input(prompt)
    await check_for_audio()
    print("Finished")

if __name__ == '__main__':
    main()
    asyncio.run(process_input_output())
    