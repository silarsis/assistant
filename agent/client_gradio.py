import gradio as gr

import os
import requests
import asyncio
import time
import re

from urllib.parse import quote

import google_auth_oauthlib

import pyaudio
import wave
import elevenlabs
from openai import OpenAI

from models.guide import Guide, DEFAULT_SESSION_ID

from transformers import pipeline

from pydantic import BaseModel
from pydantic_settings import BaseSettings
from pydantic.types import Any
import numpy as np
from numpy.typing import ArrayLike

from config import settings

# def google_logout():
#     if "_credentials" in st.session_state:
#         st.session_state._credentials = None
#         response = st.session_state.agent.bot.update_google_docs_token(st.session_state._credentials)
#         with st.chat_message("assistant"):
#             st.markdown(response.mesg)

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

class Agent(BaseModel):
    _character: str = ""
    _agent: Guide = None
    _audio_model: Any = None
    speech_engine: str = settings.voice
    _google_credentials: Any = None
    
    def __init__(self, **kwargs):
        super().__init__()
        self._connect(character=kwargs.get('character', None))
        
    def _connect(self, character: str = None):
        if character:
            self._character = character
        else:
            if os.path.exists('./character.txt'):
                filename = './character.txt'
            elif os.path.exists('./agent/character.txt'):
                filename = './agent/character.txt'
            with open(filename, 'r') as char_file:
                self._character = char_file.read()
        self._agent = Guide(default_character=self._character)
        
    async def process_audio(self, audio_state, audio_data: tuple[int, ArrayLike]):
        sample_rate, audio = audio_data
        if not audio.any():
            return [audio_state, '']
        smoothed_audio = audio.astype(np.float32)/np.max(np.abs(audio))
        if audio_state is not None:
            audio_state['audio'] = np.concatenate([audio_state, smoothed_audio])
        else:
            audio_state = {'last_seen': time.time(), 'text': '', 'audio': smoothed_audio}
        text = transcriber({"sampling_rate": sample_rate, "raw": audio_state})["text"]
        # Maintaining the last time we saw a change to the transcribed text, so we can time out the question
        if audio_state['text'] != text:
            audio_state['last_seen'] = time.time()
        audio_state['text'] = text
        if time.time() - audio_state['last_seen'] > 2:
            # We're done listening, time to process
            pass
        return [audio_state, text]
        
    async def process_file_upload(self, file_data: str) -> str:
        # Import the file, break it to it's constituent parts, save it in ChromaDB along with the other Google Docs.
        # Need to refactor chromadb out of the gdocs plugin for general use, and build a retrieval plugin.
        return
        
    async def process_input(self, input: str, history: list[list[str, str]], *args, **kwargs):
        history.append([input, "Thinking..."])
        yield(["", history, None, None])
        recvQ = asyncio.Queue()
        asyncio.create_task(self._agent.prompt_with_callback(input, callback=recvQ.put, session_id=DEFAULT_SESSION_ID, hear_thoughts=False))
        history[-1][1] = ''
        while response := await recvQ.get():
            history[-1][1] += response.mesg
            yield(["", history] + list(self.speak(response.mesg)))
            if response.final:
                break
    
    async def recv_from_bot(self, response: str):
        print(f"Received via recv_from_bot callback: {response}")
        self._recvQ.put(response)
        # Need to work out how to send this to the chat window
        
    def set_speech_engine(self, value: str) -> str:
        settings.voice = value
        return value
    
    def _clean_text_for_speech(self, text: str) -> str:
        # Clean up the text for speech
        # First, remove any markdown images
        md_img = re.compile(r'!\[.*\]\((.*)\)')
        text = md_img.sub('', text)
        # Now remove any code blocks
        code_block = re.compile(r'```.*```')
        text = code_block.sub('', text)
        return text
        
    def speak(self, payload: str = ''):
        payload = self._clean_text_for_speech(payload)
        if settings.voice == 'None':
            print("No speech engine")
            return (None, None)
        if settings.voice == 'ElevenLabs':
            print("Elevenlabs TTS")
            return (None, b"".join([bytes(a) for a in self.elevenlabs_get_stream(text=payload)]))
        elif settings.voice == 'OpenAI':
            print("OpenAI TTS")
            client = OpenAI()
            response = client.audio.speech.create(model='tts-1', voice='nova', input=payload)
            retval = (None, b''.join(response.iter_bytes()))
            return retval
        elif settings.voice == 'TTS':
            print("Local TTS")
            try:
                return (self.tts_get_stream(payload), None)
            except requests.exceptions.ConnectionError as e:
                print(f"Speech error, not speaking: {e}")
                return (None, None)
        else:
            print(f"Unknown speech engine {settings.voice}")
            return (None, None)
            
    def tts_get_stream(self, text: str) -> bytes:
        q_text = quote(text)
        print("Requesting TTS from Local TTS instance")
        with requests.get(f"http://{settings.tts_host}:{settings.tts_port}/api/tts?text={q_text}&speaker_id=p364&style_wav=&language_id=", stream=True) as wav:
            p = pyaudio.PyAudio()
            try:
                wf = wave.Wave_read(wav.raw)
            except wave.Error as e:
                print(e)
                wf = None
            if wf:
                stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                channels=1,
                                rate=22050,
                                output=True)
                while len(data := wf.readframes(1024)):
                    stream.write(data)
                stream.stop_stream()
                stream.close()
                p.terminate()
            
    def elevenlabs_get_stream(self, text: str = '') -> bytes:
        elevenlabs.set_api_key(settings.elevenlabs_api_key)
        voices = [settings.elevenlabs_voice_1_id, settings.elevenlabs_voice_2_id]
        try:
            audio_stream = elevenlabs.generate(text=text, voice=voices[0], stream=True)
        except elevenlabs.api.error.RateLimitError as err:
            print(str(err), flush=True)
        return audio_stream
    
    def google_login(self, btn: str) -> str:
        # Gets user credentials for Google OAuth and sends them to the websocket
        # connection to authorize access to Google Drive and Docs APIs.
        if btn != "Google Login":
            print("Ignoring further button presses")
            return btn
        self._google_credentials = google_auth_oauthlib.get_user_credentials(
            ["https://www.googleapis.com/auth/drive.readonly", "https://www.googleapis.com/auth/documents.readonly"],
            # app desktop credentials
            "438635256773-rf4rmv51lo436a576enb74t7pc9n8rre.apps.googleusercontent.com",
            "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",
        )
        response = self._agent.update_google_docs_token(self._google_credentials)
        return response.mesg
    
    def update_api_keys(self, api_type: str, api_key: str, api_base: str, deployment_name: str) -> str:
        settings.openai_api_type = api_type
        settings.openai_api_key = api_key
        settings.openai_api_base = api_base
        settings.openai_deployment_name = deployment_name
        # Reconnect to OpenAI here - chance to update the character also
        self._connect(character=self._character)
        return [api_type, api_key, api_base, deployment_name]
    
    def update_character(self, character: str) -> str:
        self._character = character
        # Reconnect to OpenAI here - chance to update the character also
        self._connect(character=character)
        return character

agent = Agent()

with gr.Blocks() as demo:
    with gr.Accordion("Settings", open=False):
        speech_engine = gr.Dropdown(["None", "ElevenLabs", "OpenAI", "TTS"], label="Speech Engine", value=settings.voice, interactive=True)
        speech_engine.input(agent.set_speech_engine, [speech_engine], [speech_engine])
        google_login_button = gr.Button("Google Login")
        google_login_button.click(agent.google_login, [google_login_button], [google_login_button])
        with gr.Accordion("OpenAI Keys", open=False):
            api_config = [
                gr.Dropdown(["openai", "azure"], label="API Type", value=settings.openai_api_type, interactive=True),
                gr.Textbox(label="API Key", value=settings.openai_api_key, type="password"),
                gr.Textbox(label="API Base URI", value=settings.openai_api_base, type="text"),
                gr.Textbox(label="Deployment Name", value=settings.openai_deployment_name, type="text"),
            ]
            api_update_button = gr.Button("Update")
            api_update_button.click(agent.update_api_keys, api_config, api_config)
        with gr.Accordion("Character", open=False):
            char = gr.Textbox(agent._character, show_copy_button=True, lines=5)
            char_btn = gr.Button("Update")
            char_btn.click(agent.update_character, [char], [char])
    with gr.Row():
        wav_speaker = gr.Audio(interactive=False, streaming=True, visible=False, format='wav', autoplay=True)
        mp3_speaker = gr.Audio(interactive=False, visible=False, format='mp3', autoplay=True)
    chatbot = gr.Chatbot([], bubble_full_width=False)
    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter",
            container=False,
        )
        txt.submit(agent.process_input, [txt, chatbot], [txt, chatbot, wav_speaker, mp3_speaker])
        # btn = gr.UploadButton("📁", type="binary")
        # btn.upload(agent.process_file_upload, [btn], [chatbot])
    with gr.Row():
        audio_state = gr.State()
        audio = gr.Audio(sources="microphone", streaming=True, autoplay=True)
        audio.stream(agent.process_audio, [audio_state, audio], [audio_state, txt])
        audio.start_recording(lambda x:None, [audio_state], [audio_state]) # This wipes the audio_state at the start of listening
        audio.stop_recording(agent.process_input, [txt, chatbot], [txt, chatbot, wav_speaker, mp3_speaker])

demo.queue()
if __name__ == '__main__':
    demo.launch()