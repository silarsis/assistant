import gradio as gr

import os
import requests
import asyncio
import time

from urllib.parse import quote

# import google_auth_oauthlib

import pyaudio
import wave
import elevenlabs
from openai import OpenAI

from models.guide import Guide, DEFAULT_SESSION_ID

from transformers import pipeline

from pydantic import BaseModel
from pydantic.types import Any
import numpy as np
from numpy.typing import ArrayLike


TTS_HOST=os.environ.get("TTS_HOST", "localhost")
TTS_PORT=os.environ.get("TTS_PORT", "5002")
        

# def google_login():
#     # Gets user credentials for Google OAuth and sends them to the websocket
#     # connection to authorize access to Google Drive and Docs APIs.
#     if "_credentials" not in st.session_state:
#         st.session_state._credentials = google_auth_oauthlib.get_user_credentials(
#             ["https://www.googleapis.com/auth/drive.readonly", "https://www.googleapis.com/auth/documents.readonly"],
#             # app desktop credentials
#             "438635256773-rf4rmv51lo436a576enb74t7pc9n8rre.apps.googleusercontent.com",
#             "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",
#         )
#         response = st.session_state.agent.bot.update_google_docs_token(st.session_state._credentials)
#         with st.chat_message("assistant"):
#             st.markdown(response.mesg)
            
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
    speech_engine: str = os.environ.get('VOICE', 'None')
    
    def __init__(self, **kwargs):
        super().__init__()
        if 'character' in kwargs:
            self._character = kwargs['character']
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
        self.speech_engine = value
        return value
        
    def speak(self, payload: str = ''):
        if self.speech_engine == 'None':
            print("No speech engine")
            return (None, None)
        if self.speech_engine == 'ElevenLabs':
            print("Elevenlabs TTS")
            return (None, b"".join([bytes(a) for a in self.elevenlabs_get_stream(text=payload)]))
        elif self.speech_engine == 'OpenAI':
            print("OpenAI TTS")
            client = OpenAI()
            response = client.audio.speech.create(model='tts-1', voice='nova', input=payload)
            retval = (None, b''.join(response.iter_bytes()))
            return retval
        elif self.speech_engine == 'TTS':
            print("Local TTS")
            try:
                return (self.tts_get_stream(payload), None)
            except requests.exceptions.ConnectionError as e:
                print(f"Speech error, not speaking: {e}")
                return (None, None)
        else:
            print(f"Unknown speech engine {self.speech_engine}")
            return (None, None)
            
    def tts_get_stream(self, text: str) -> bytes:
        q_text = quote(text)
        print("Requesting TTS from Local TTS instance")
        with requests.get(f"http://{TTS_HOST}:{TTS_PORT}/api/tts?text={q_text}&speaker_id=p364&style_wav=&language_id=", stream=True) as wav:
            p = pyaudio.PyAudio()
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
            
    def elevenlabs_get_stream(self, text: str = '') -> bytes:
        elevenlabs.set_api_key(os.environ.get('ELEVENLABS_API_KEY'))
        voices = [os.environ.get('ELEVENLABS_VOICE_1_ID'), os.environ.get('ELEVENLABS_VOICE_2_ID')]
        try:
            audio_stream = elevenlabs.generate(text=text, voice=voices[0], stream=True)
        except elevenlabs.api.error.RateLimitError as err:
            print(str(err), flush=True)
        return audio_stream

agent = Agent()

with gr.Blocks() as demo:
    with gr.Accordion("Settings", open=True):
        speech_engine = gr.Dropdown(["None", "ElevenLabs", "OpenAI", "TTS"], label="Speech Engine", value=os.environ.get('VOICE', 'None'), interactive=True)
        speech_engine.input(agent.set_speech_engine, [speech_engine], [speech_engine])
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