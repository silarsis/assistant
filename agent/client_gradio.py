import gradio as gr

import os
import requests
import asyncio
import time
import re
import functools
from io import StringIO
from markdown import Markdown
from models.tools.clean_markdown import convert_to_plain_text

from urllib.parse import quote

import google_auth_oauthlib

import pyaudio
import wave
import elevenlabs

from models.tools.llm_connect import LLMConnect

from models.guide import Guide, DEFAULT_SESSION_ID

try:
    from transformers import pipeline
except (ImportError, RuntimeError): # RuntimeError for a circular import
    pipeline = None

from pydantic import BaseModel
from pydantic.types import Any, Union, List
import numpy as np
from numpy.typing import ArrayLike

from config import settings, AgentModel

# def google_logout():
#     if "_credentials" in st.session_state:
#         st.session_state._credentials = None
#         response = st.session_state.agent.bot.update_google_docs_token(st.session_state._credentials)
#         with st.chat_message("assistant"):
#             st.markdown(response.mesg)

if pipeline:
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

@functools.lru_cache()
def elevenlabs_voices():
    return [[voice.name, voice.voice_id] for voice in elevenlabs.voices()]

# From https://stackoverflow.com/questions/761824/python-how-to-convert-markdown-formatted-text-to-text,
# code to turn markdown into plain text so it can be read nicely
def unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()
Markdown.output_formats["plain"] = unmark_element


class Agent(BaseModel):
    _character: str = ""
    _agent: Guide = None
    _audio_model: Any = None
    speech_engine: str = settings.voice
    _google_credentials: Any = None
    session_id: str = DEFAULT_SESSION_ID
    settings_block: Any = None
    
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
        
    def update_session_id(self, session_id: str) -> str:
        self.session_id = session_id
        return self.session_id
        
    async def process_audio(self, audio_state, audio_data: tuple[int, ArrayLike]):
        if not transcriber:
            print("No transcriber available, skipping audio processing")
            return [None, '']
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
    
    async def process_file_input(self, filename: str, history: list[list[str, str]], *args, **kwargs):
        history.append((filename, "Thinking..."))
        yield([history, None, None])
        recvQ = asyncio.Queue()
        async with asyncio.TaskGroup() as tg:
            # Assigned to a variable to keep it in scope so the task doesn't get deleted too early
            prompt_task = tg.create_task(
                self._agent.prompt_file_with_callback(
                    filename, callback=recvQ.put, session_id=self.session_id, hear_thoughts=False), name="prompt")
            history[-1][1] = ''
            while response := await recvQ.get():
                history[-1][1] += response.mesg
                yield([history] + list(self.speak(response.mesg)))
                if response.final: # Could also check here if the task is complete?
                    break
        
    async def process_input(self, input: Union[str,bytes], history: list[list[str, str]], *args, **kwargs):
        history.append([input, "Thinking..."])
        yield(["", history, None, None])
        recvQ = asyncio.Queue()
        async with asyncio.TaskGroup() as tg:
            # Assigned to a variable to keep it in scope so the task doesn't get deleted too early
            prompt_task = tg.create_task(
                self._agent.prompt_with_callback(
                    input, callback=recvQ.put, session_id=self.session_id, hear_thoughts=False), name="prompt")
            history[-1][1] = ''
            while response := await recvQ.get():
                history[-1][1] += response.mesg
                yield(["", history] + list(self.speak(response.mesg)))
                if response.final: # Could also check here if the task is complete?
                    break
    
    async def recv_from_bot(self, response: str):
        print(f"Received via recv_from_bot callback: {response}")
        self._recvQ.put(response)
        # Need to work out how to send this to the chat window
        
    def set_speech_engine(self, value: str) -> str:
        settings.voice = value
        settings.save()
        return value
    
    def _clean_text_for_speech(self, text: str) -> str:
        # Clean up the text for speech
        # First, remove any markdown images
        cleaned = convert_to_plain_text(text)
        # md_img = re.compile(r'!\[.*\]\((.*)\)')
        # text = md_img.sub('', text)
        # Now remove any code blocks
        code_block = re.compile(r'```(?:.|\n)*?```')
        cleaned = code_block.sub("(I try not to read code aloud, but check the screen)", cleaned)
        return text
        
    def speak(self, payload: str = ''):
        payload = self._clean_text_for_speech(payload)
        if not payload:
            print("Nothing to say")
            return (None, None)
        if settings.voice == 'None':
            print("No TTS")
            return (None, None)
        if settings.voice == 'ElevenLabs':
            print("Elevenlabs TTS")
            return (None, b"".join([bytes(a) for a in self.elevenlabs_get_stream(text=payload)]))
        elif settings.voice == 'OpenAI':
            print("OpenAI TTS")
            client = LLMConnect(
                api_type=settings.openai_api_type, 
                api_key=settings.openai_api_key, 
                api_base=settings.openai_api_base, 
                deployment_name=settings.openai_deployment_name, 
                org_id=settings.openai_org_id
            ).openai()
            # Break into max(4096) character chunks here, split by word
            if len(payload) > 4096:
                payload = "Sorry, this text was too long, you'll have to read it instead"
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
            print(f"Unknown TTS {settings.voice}")
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
    
    def update_api_keys(self, api_type: str, api_key: str, api_base: str, deployment_name: str, org_id: str) -> str:
        settings.openai_api_type = api_type
        settings.openai_api_key = api_key
        settings.openai_api_base = api_base or None
        settings.openai_deployment_name = deployment_name
        settings.openai_org_id = org_id or None
        self._connect(character=self._character)
        settings.save()
        return [api_type, api_key, api_base, deployment_name, org_id]
    
    def update_img_api_keys(self, inherit: bool, api_type: str, api_key: str, api_base: str) -> List:
        settings.img_openai_inherit = inherit
        settings.img_openai_api_type = api_type
        settings.img_openai_api_key = api_key
        settings.img_openai_api_base = api_base
        settings.save()
        return [
            gr.Checkbox(label="Inherit from Main LLM", value=settings.img_openai_inherit, interactive=True),
            gr.Dropdown(["openai", "azure"], label="API Type", value=settings.img_openai_api_type, visible=(settings.img_openai_inherit is False)),
            gr.Textbox(label="API Key", value=settings.img_openai_api_key, type="password", visible=(settings.img_openai_inherit is False)),
            gr.Textbox(label="API Base URI", value=settings.img_openai_api_base, type="text", visible=(settings.img_openai_inherit is False))
        ]
    
    def update_character(self, character: str) -> str:
        self._character = character
        self._connect(character=character)
        return character
    
    def update_img_api_key(self, api_key: str) -> str:
        settings.img_upload_api_key = api_key
        settings.save()
    
    def get_history_for_chatbot(self):
        return self._agent.memory.get_history_for_chatbot(self.session_id)
    
    def update_voice_settings(self, speech_engine, el_api_key, el_voice1, tts_host, tts_port) -> List:
        settings.voice = speech_engine
        if speech_engine == 'ElevenLabs':
            settings.elevenlabs_api_key = el_api_key
            settings.elevenlabs_voice_1_id = el_voice1
        if speech_engine == 'TTS':
            settings.tts_host = tts_host
            settings.tts_port = tts_port
        settings.save()
        return [
            gr.Textbox(label="ElevenLabs API Key", value=settings.elevenlabs_api_key, type="password", visible=(settings.voice == 'ElevenLabs')),
            gr.Dropdown(elevenlabs_voices(), label="ElevenLabs Voice", value=settings.elevenlabs_voice_1_id, interactive=True, visible=(settings.voice == 'ElevenLabs')),
            gr.Textbox(label="TTS Host", value=settings.tts_host, visible=(settings.voice == 'TTS')),
            gr.Textbox(label="TTS Port", value=settings.tts_port, visible=(settings.voice == 'TTS'))
        ]
        
    def update_presto_settings(self, presto_host: str, presto_username: str, presto_password: str) -> None:
        settings.presto_host = presto_host
        settings.presto_username = presto_username
        settings.presto_password = presto_password
        settings.save()
        
    def update_crew_settings(self, crew_num: int, role: str, goal: str, backstory: str) -> None:
        crew = AgentModel(role=role, goal=goal, backstory=backstory)
        if crew_num >= len(settings.crew):
            settings.crew.append(crew)
        else:
            settings.crew[crew_num] = crew
        settings.save()

def render_crew(crew_num: int, crew: AgentModel, render: bool = True) -> gr.Accordion:
    with gr.Accordion(f"Crewmember role: {crew.role or 'New'}", open=False, render=render) as crew_member:
        crew_settings = [
            gr.Number(value=crew_num, visible=False),
            gr.Textbox(label='Role', value=crew.role),
            gr.Textbox(label='Goal', value=crew.goal),
            gr.Textbox(label='Backstory', value=crew.backstory)
        ]
        update_button = gr.Button("Update")
        update_button.click(agent.update_crew_settings, crew_settings)
    return crew_member

agent = Agent(character=settings.character)

with gr.Blocks(fill_height=True) as demo:
    agent.settings_block = demo
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Accordion("Settings", open=True):
                with gr.Row():
                    with gr.Column(scale=4):
                        google_login_button = gr.Button("Google Login")
                        google_login_button.click(agent.google_login, [google_login_button], [google_login_button])
                    with gr.Column(scale=1):
                        sesid = gr.Textbox(label="Session ID", value=DEFAULT_SESSION_ID)
                        sesid.change(agent.update_session_id, [sesid], [sesid])
                with gr.Accordion("Speech", open=False):
                    speech_engine = gr.Dropdown(["None", "ElevenLabs", "OpenAI", "TTS"], label="Speech Engine", value=settings.voice, interactive=True)
                    with gr.Row():
                        with gr.Row():
                            el_api_key = gr.Textbox(label="ElevenLabs API Key", value=settings.elevenlabs_api_key, type="password", visible=(settings.voice == 'ElevenLabs'))
                            tts_host = gr.Textbox(label="TTS Host", value=settings.tts_host, visible=(settings.voice == 'TTS'))
                        with gr.Row():
                            el_voice1 = gr.Dropdown(elevenlabs_voices(), label="ElevenLabs Voice", value=settings.elevenlabs_voice_1_id, interactive=True, visible=(settings.voice == 'ElevenLabs'))
                            tts_port = gr.Textbox(label="TTS Port", value=settings.tts_port, visible=(settings.voice == 'TTS'))
                        with gr.Row():
                            voice_params_submit = gr.Button("Update")
                            speech_engine.change(agent.update_voice_settings, [speech_engine, el_api_key, el_voice1, tts_host, tts_port], [el_api_key, el_voice1, tts_host, tts_port])
                            voice_params_submit.click(agent.update_voice_settings, [speech_engine, el_api_key, el_voice1, tts_host, tts_port], [el_api_key, el_voice1, tts_host, tts_port])
                with gr.Accordion("Main LLM Keys", open=False):
                    api_config = [
                        gr.Dropdown(["openai", "azure"], label="API Type", value=settings.openai_api_type, interactive=True),
                        gr.Textbox(label="API Key", value=settings.openai_api_key, type="password"),
                        gr.Textbox(label="API Base URI", value=settings.openai_api_base, type="text"),
                        gr.Textbox(label="Deployment Name", value=settings.openai_deployment_name, type="text"),
                        gr.Textbox(label="Org ID", value=settings.openai_org_id, type="text"),
                    ]
                    api_update_button = gr.Button("Update")
                    api_update_button.click(agent.update_api_keys, api_config, api_config)
                with gr.Accordion("Image Generation Keys", open=False):
                    api_config = [
                        gr.Checkbox(label="Inherit from Main LLM", value=settings.img_openai_inherit, interactive=True),
                        gr.Dropdown(["openai", "azure"], label="API Type", value=settings.img_openai_api_type, visible=(settings.img_openai_inherit is False)),
                        gr.Textbox(label="API Key", value=settings.img_openai_api_key, type="password", visible=(settings.img_openai_inherit is False)),
                        gr.Textbox(label="API Base URI", value=settings.img_openai_api_base, type="text", visible=(settings.img_openai_inherit is False))
                    ]
                    api_update_button = gr.Button("Update")
                    api_update_button.click(agent.update_img_api_keys, api_config, api_config)
                    api_config[0].change(agent.update_img_api_keys, api_config, api_config)
                with gr.Accordion("Image Upload Keys", open=False):
                    api_key = gr.Textbox(label="Image API Key", value=settings.img_upload_api_key, type="password")
                    api_key.input(agent.update_img_api_key, [api_key])
                with gr.Accordion("Presto", open=False):
                    presto_config = [
                        gr.Textbox(label="Presto Hostname", value=settings.presto_host),
                        gr.Textbox(label="Presto Username", value=settings.presto_username),
                        gr.Textbox(label="Presto Password", value=settings.presto_password, type="password")
                    ]
                    presto_button = gr.Button("Update")
                    presto_button.click(agent.update_presto_settings, presto_config)
            with gr.Row():
                wav_speaker = gr.Audio(interactive=False, streaming=True, visible=False, format='wav', autoplay=True)
                mp3_speaker = gr.Audio(interactive=False, visible=False, format='mp3', autoplay=True)
        with gr.Column(scale=8):
            with gr.Tab('ChatBot'):
                chatbot = gr.Chatbot(agent.get_history_for_chatbot, bubble_full_width=False, show_copy_button=True)
                with gr.Row():
                    txt = gr.Textbox(
                        scale=4,
                        show_label=False,
                        placeholder="Enter text and press enter",
                        container=False,
                    )
                    txt.submit(agent.process_input, [txt, chatbot], [txt, chatbot, wav_speaker, mp3_speaker])
                    btn = gr.UploadButton("📁", type="filepath")
                    btn.upload(agent.process_file_input, [btn, chatbot], [chatbot, wav_speaker, mp3_speaker])
                with gr.Row():
                    audio_state = gr.State()
                    audio = gr.Audio(sources="microphone", streaming=True, autoplay=True)
                    audio.stream(agent.process_audio, [audio_state, audio], [audio_state, txt])
                    audio.start_recording(lambda x:None, [audio_state], [audio_state]) # This wipes the audio_state at the start of listening
                    audio.stop_recording(agent.process_input, [txt, chatbot], [txt, chatbot, wav_speaker, mp3_speaker])
            with gr.Tab('Character'):
                char = gr.Textbox(agent._character, show_copy_button=True, lines=15)
                char_btn = gr.Button("Update")
                char_btn.click(agent.update_character, [char], [char])
            with gr.Tab('Crew') as crew_tab:
                all_crew =[]
                for crew_num, crew in enumerate(settings.crew):
                    all_crew.append(render_crew(crew_num, crew))
                for new_crew_num in range(len(all_crew), 5):
                    all_crew.append(render_crew(new_crew_num, AgentModel(role='', goal='', backstory='')))
                

demo.queue()
if __name__ == '__main__':
    demo.launch()