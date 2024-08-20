import gradio as gr

import requests
import asyncio
import time
import re
import functools
import hashlib
import base64
import secrets
import string
import os
import tempfile
import urllib.parse
from io import StringIO
from markdown import Markdown
import json
from asyncio import QueueEmpty
from collections.abc import Iterable
from typing import Annotated

from models.tools.clean_markdown import convert_to_plain_text
from models.tools.radio import AIRadio

from urllib.parse import quote

import google_auth_oauthlib

import pyaudio
import wave
from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError

from models.tools.llm_connect import LLMConnect
from models.tools.memory import Memory

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

HistoryType = list[list[str, str]]

SpotifyScript = """
window.onSpotifyWebPlaybackSDKReady = () => {
  const token = '{settings.spotify_token}';
  const player = new Spotify.Player({
    name: 'Web Playback SDK Quick Start Player',
    getOAuthToken: cb => { cb(token); },
    volume: 0.5
  });
  player.connect();
"""

def generate_code_verifier(length=128):
    """
    Generates a 'code_verifier' which is a high-entropy cryptographic random STRING using the
    unreserved characters [A-Z] / [a-z] / [0-9] / "-" / "." / "_" / "~" from Section 2.3 of [RFC3986],
    with a minimum length of 43 characters and a maximum length of 128 characters.
    """
    if settings.spotify_client_verifier: # Cacheing this is probably bad.
        return settings.spotify_client_verifier
    token = ''.join([ secrets.choice(string.ascii_letters + string.digits + '_.-~') for i in range(length) ])
    settings.spotify_client_verifier = token
    settings.save()
    return token
    # # Generate random bytes and convert them into a base64 URL-encoded string.
    # token = os.urandom(length)
    # code_verifier = base64.urlsafe_b64encode(token).decode('utf-8').rstrip('=')
    # return code_verifier[:length]

def generate_code_challenge(code_verifier):
    """
    Generates a 'code_challenge' derived from the code verifier by using SHA256 hashing and then
    base64 URL-encoding the result. The transformation of the 'code_verifier' to the 'code_challenge'
    is referred to as code_challenge_method and the method used is S256.
    """
    sha256 = hashlib.sha256(code_verifier.encode('ascii')).digest()
    return base64.urlsafe_b64encode(sha256).decode('ascii').rstrip('=')
    # # SHA256 hash the code_verifier and then base64 URL-encode the result.
    # sha256 = hashlib.sha256(code_verifier.encode('utf-8')).digest()
    # code_challenge = base64.urlsafe_b64encode(sha256).decode('utf-8').rstrip('=')
    # return code_challenge


# def google_logout():
#     if "_credentials" in st.session_state:
#         st.session_state._credentials = None
#         response = st.session_state.agent.bot.update_google_docs_token(st.session_state._credentials)
#         with st.chat_message("assistant"):
#             st.markdown(response.mesg)

if pipeline:
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

@functools.lru_cache()
def elevenlabs_voices(): # TODO: Refresh this when the API key is changed
    client = ElevenLabs(api_key=settings.elevenlabs_api_key)
    try:
        voices = client.voices.get_all().voices
    except ApiError as e:
        if e.status_code == 401: # Invalid API key
            voices = []
        voices = []
    return [[voice.name, voice.voice_id] for voice in voices]

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
    character: str = ""
    _agent: Guide = None
    _audio_model: Any = None
    speech_engine: str = settings.voice
    _google_credentials: Any = None
    session_id: str = DEFAULT_SESSION_ID
    settings_block: Any = None
    _radio: AIRadio = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._radio = AIRadio()
        self._connect()

    def _connect(self):
        self._agent = Guide(default_character=self.character)

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

    async def process_file_input_memories(self, filename: str, *args, **kwargs):
        response = await self._agent.remember_file(filename)
        # Can we alert with the response here?
        gr.Info(response.mesg)
        return [ fn for dn, fn in self.documents() if dn == os.path.basename(filename) ][0]

    async def process_file_input_inline(self, filename: str, history: HistoryType, *args, **kwargs):
        base_filename = os.path.basename(filename)
        history.append(gr.ChatMessage(role="user", content=base_filename))
        history.append(gr.ChatMessage(role="assistant", content="Thinking..."))
        yield([history, None, None])
        recvQ = asyncio.Queue()
        async with asyncio.TaskGroup() as tg:
            # Assigned to a variable to keep it in scope so the task doesn't get deleted too early
            _prompt_task = tg.create_task(
                self._agent.prompt_file_with_callback(
                    filename, callback=recvQ.put, session_id=self.session_id, hear_thoughts=settings.hear_thoughts), name="prompt")
            history[-1].content = ""
            while response := await recvQ.get():
                if response.final:
                    async for result in self.process_input(response.mesg, history, *args, **kwargs):
                        yield result
                    break
                history[-1].content += response.mesg
                yield([history] + list(self.speak(response.mesg)))


    async def process_input(self, input: Union[str, bytes], history: HistoryType, *args, **kwargs):
        history.append(gr.ChatMessage(role="user", content=input))
        history.append(gr.ChatMessage(role="assistant", content="Thinking..."))
        yield([history, None, None])
        recvQ = asyncio.Queue()
        async with asyncio.TaskGroup() as tg:
            # Assigned to a variable to keep it in scope so the task doesn't get deleted too early
            _prompt_task = tg.create_task(
                self._agent.prompt_with_callback(
                    input, callback=recvQ.put, session_id=self.session_id, hear_thoughts=settings.hear_thoughts), name="prompt")
            history[-1].content = ''
            while response := await recvQ.get():
                history[-1].content += response.mesg
                yield([history] + list(self.speak(response.mesg)))
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
        with requests.get(f"http://{settings.tts_host}:{settings.tts_port}/api/tts?text={q_text}&speaker_id=p364&style_wav=&language_id=", stream=True, timeout=10) as wav:
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
        client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        voices = [settings.elevenlabs_voice_1_id, settings.elevenlabs_voice_2_id]
        try:
            audio_stream = client.generate(text=text, voice=voices[0], stream=True)
        except Exception as err:
            print(str(err), flush=True)
        return audio_stream

    def google_login(self, btn: str) -> str:
        # Gets user credentials for Google OAuth and sends them to the agent to authorize access to Google Drive and Docs APIs.
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

    def list_plugins(self):
        return self._agent.list_plugins()

    def update_api_keys(self, api_type: str, api_key: str, api_base: str, deployment_name: str, org_id: str) -> str:
        settings.openai_api_type = api_type
        settings.openai_api_key = api_key
        settings.openai_api_base = api_base or None
        settings.openai_deployment_name = deployment_name
        settings.openai_org_id = org_id or None
        self._connect()
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
        self.character = character
        self._connect()
        return character

    def update_img_api_key(self, api_key: str) -> str:
        settings.img_upload_api_key = api_key
        settings.save()

    def get_history_for_chatbot(self) -> list[gr.ChatMessage]:
        return [
            gr.ChatMessage(role=mesg['role'], content=mesg['content'])
            for mesg in Memory(session_id=self.session_id).get_history_for_chatbot()
        ]

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

    def update_presto_settings(self, presto_host: str, presto_username: str, presto_password: str) -> list[str, str, str]:
        settings.presto_host = presto_host
        settings.presto_username = presto_username
        settings.presto_password = presto_password
        settings.save()
        return presto_host, presto_username, presto_password

    def update_confluence_settings(self, confluence_uri: str, confluence_pat: str) -> list[str, str]:
        settings.confluence_uri = confluence_uri
        settings.confluence_pat = confluence_pat
        settings.save()
        return confluence_uri, confluence_pat

    def update_crew_settings(self, crew_num: int, role: str, goal: str, backstory: str) -> list[int, str, str, str]:
        crew = AgentModel(role=role, goal=goal, backstory=backstory)
        if crew_num >= len(settings.crew):
            settings.crew.append(crew)
        else:
            settings.crew[crew_num] = crew
        settings.save()
        return crew_num, role, goal, backstory

    def update_client_settings(self, hear_thoughts: bool) -> bool:
        settings.hear_thoughts = hear_thoughts
        settings.save()
        return hear_thoughts

    def update_radio(self, prompt: str, history: HistoryType) -> list[str, HistoryType]:
        history.append(gr.ChatMessage(role="user", content="prompt"))
        history.append(gr.ChatMessage(role="assistant", content="Thinking..."))
        return [prompt, history]

    async def spotify_login_button(self, code: str, state: str = '') -> str:
        settings.spotify_client_code = code
        params = {
            'grant_type': 'authorization_code',
            'code': settings.spotify_client_code,
            'redirect_uri': 'http://localhost:7860/spotify_login',
            'client_id': settings.spotify_client_id,
            'code_verifier': settings.spotify_client_verifier
        }
        auth_response = requests.post("https://accounts.spotify.com/api/token", params=params, headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=10)
        # Shouldn't store these in the config, should go to the spotify tool
        settings.spotify_access_token = auth_response.json()['access_token']
        settings.spotify_refresh_token = auth_response.json()['refresh_token']
        settings.spotify_expiry = auth_response.json()['expires_in'] + time.time() - 5 # 5 seconds before for some buffer
        settings.save()
        return "Logged in - please return to your main page"

    def play_pause_radio(self, button: str) -> List:
        if button == "▶️":
            self._agent._radio.play()
            button = '⏸️'
        elif button == "⏸️":
            self._agent._radio.pause()
            button = '▶️'
        return button

    def documents(self) -> Iterable[Annotated[str, "docname"], Annotated[str, "full pathname for file"]]:
        return self._agent.documents()

    def delete_document(self, filepaths: str) -> None:
        " Takes a list of files that exist, deletes the rest "
        filenames = [ os.path.basename(filepath) for filepath in filepaths ]
        for docname, _pathname in self._agent.documents():
            if docname not in filenames:
                self._agent.delete_document(docname)

    async def search_wiki(self, cql: str, history: HistoryType = []) -> str:
        uri = urllib.parse.urljoin(settings.confluence_uri, "wiki/rest/api/content/search")
        results = requests.get(
            uri, params={'cql': cql},
            headers={'Authorization': f"Bearer {settings.confluence_pat}"}, timeout=10)
        # Get the results of the request, save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tfile:
            tfile.write(results.text)
            filename = tfile.name
        input_result = self.process_file_input_memories(filename)
        os.unlink(filename)
        return input_result

def render_crew(crew_num: int, crew: AgentModel, render: bool = True) -> gr.Accordion:
    with gr.Accordion(f"Crewmember role: {crew.role or 'New'}", open=False, render=render) as crew_member:
        crew_settings = [
            gr.Number(value=crew_num, visible=False),
            gr.Textbox(label='Role', value=crew.role),
            gr.Textbox(label='Goal', value=crew.goal),
            gr.Textbox(label='Backstory', value=crew.backstory, lines=5)
        ]
        update_button = gr.Button("Update")
        update_button.click(agent.update_crew_settings, crew_settings, crew_settings)
    return crew_member

async def generate_crew(goal: str = "", *crew_fields) -> List[str]:
    if goal:
        # Call the LLM to generate the crew, parse the results and create each of the crew members
        crew = await agent._agent.generate_crew(goal)
        try:
            new_crew = json.loads(str(crew))['crew']
        except Exception:
            new_crew = []
        crew_fields = list(crew_fields)
        for crew_num, fields in enumerate(range(0, len(crew_fields), 3)):
            if crew_num >= len(new_crew):
                new_crew.append({'role':'', 'goal':'', 'backstory':''})
            crew_fields[fields] = new_crew[crew_num]['role']
            crew_fields[fields+1] = new_crew[crew_num]['goal']
            crew_fields[fields+2] = new_crew[crew_num]['backstory']
            agent.update_crew_settings(crew_num, new_crew[crew_num]['role'], new_crew[crew_num]['goal'], new_crew[crew_num]['backstory'])
        settings.save()
    return crew_fields

agent = Agent(character=settings.character)

def spotify_link():
    settings.spotify_client_verifier = generate_code_verifier()
    params = {
        'client_id': settings.spotify_client_id,
        'response_type': 'code',
        'redirect_uri': 'http://localhost:7860/spotify_login',
        'scope': 'user-modify-playback-state',
        'code_challenge_method': 'S256',
        'code_challenge': generate_code_challenge(settings.spotify_client_verifier)
    }
    return "https://accounts.spotify.com/authorize?" + requests.compat.urlencode(params)

def setup_spotify_login(button: gr.Button):
    demo.app.add_api_route('/spotify_login', agent.spotify_login_button, methods=['GET'])

def radio_tick_mp3(*args, **kwargs):
    try:
        if mp3 := agent._radio._mp3_queue.get_nowait():
            if isinstance(mp3, str):
                agent.speak(mp3)
                return None
            return [mp3]
    except QueueEmpty:
        return None

def radio_tick_wav(*args, **kwargs):
    try:
        if wav := agent._radio._wav_queue.get_nowait():
            return [wav]
    except QueueEmpty:
        return None

with gr.Blocks(fill_height=True, head='<script src="https://sdk.scdn.co/spotify-player.js"></script>') as demo:
    agent.settings_block = demo
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Accordion("Settings", open=True):
                with gr.Accordion("Client Settings", open=False):
                    with gr.Row():
                        google_login_button = gr.Button("Google Login")
                        google_login_button.click(agent.google_login, [google_login_button], [google_login_button])
                    with gr.Row():
                        spotify_login_button = gr.Button("Spotify Login", link=spotify_link())
                        demo.load(setup_spotify_login, inputs=[spotify_login_button])
                    with gr.Row():
                        sesid = gr.Textbox(label="Session ID", value=DEFAULT_SESSION_ID)
                        sesid.change(agent.update_session_id, [sesid], [sesid])
                    with gr.Row():
                        hear_thoughts = gr.Checkbox(label="Hear Thoughts", value=settings.hear_thoughts, interactive=True)
                        hear_thoughts.change(agent.update_client_settings, [hear_thoughts], [hear_thoughts])
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
                with gr.Accordion("Confluence", open=False):
                    confluence_config = [
                        gr.Textbox(label="Confluence URI", value=settings.confluence_uri),
                        gr.Textbox(label="Confluence Personal Access Token", value=settings.confluence_pat, type="password")
                    ]
                    gr.Button("Update").click(agent.update_confluence_settings, confluence_config, confluence_config)
            with gr.Row():
                wav_speaker = gr.Audio(interactive=False, streaming=True, visible=False, format='wav', autoplay=True)
                mp3_speaker = gr.Audio(interactive=False, visible=False, format='mp3', autoplay=True)
        with gr.Column(scale=8):
            with gr.Tab('ChatBot'):
                chatbot = gr.Chatbot(agent.get_history_for_chatbot, bubble_full_width=False, show_copy_button=True, height="80vh", type="messages")
                with gr.Row():
                    txt = gr.Textbox(
                        scale=8,
                        show_label=False,
                        placeholder="Enter text and press enter",
                        container=False,
                    )
                    txt.submit(agent.process_input, [txt, chatbot], [chatbot, wav_speaker, mp3_speaker])
                    submit_btn = gr.Button("▶️", scale=1)
                    submit_btn.click(agent.process_input, [txt, chatbot], [chatbot, wav_speaker, mp3_speaker])
                    btn = gr.UploadButton("📁", type="filepath", scale=1)
                    btn.upload(agent.process_file_input_inline, [btn, chatbot], [chatbot, wav_speaker, mp3_speaker])
                # with gr.Row():
                #     audio_state = gr.State()
                #     audio = gr.Audio(sources="microphone", streaming=True, autoplay=True)
                #     audio.stream(agent.process_audio, [audio_state, audio], [audio_state, txt])
                #     audio.start_recording(lambda x:None, [audio_state], [audio_state]) # This wipes the audio_state at the start of listening
                #     audio.stop_recording(agent.process_input, [txt, chatbot], [txt, chatbot, wav_speaker, mp3_speaker])
            with gr.Tab('Character'):
                char = gr.Textbox(agent.character, show_copy_button=True, lines=15)
                char_btn = gr.Button("Update")
                char_btn.click(agent.update_character, [char], [char])
            with gr.Tab('Tools'):
                for plugin_name, plugin in agent.list_plugins().items():
                    with gr.Accordion(plugin_name, open=False):
                        for f_name in plugin.functions:
                            gr.Checkbox(value=True, label=f"{plugin_name}.{f_name}", interactive=True)
            with gr.Tab('Radio') as radio_tab:
                # In here somewhere, need to include https://developer.spotify.com/documentation/web-playback-sdk/tutorials/getting-started
                textboxes = [
                    gr.Textbox(label="Prompt", placeholder="80's and 90's greatest hits and influential music", type="text"),
                    gr.Textbox(label="Announcer Style", placeholder="Casey Kasem's American Top 40", type="text")
                ]
                with gr.Row():
                    wav_radio = gr.Audio(interactive=False, streaming=True, visible=False, format='wav', autoplay=True)#, every=1, value=radio_tick_wav)
                    mp3_radio = gr.Audio(interactive=False, visible=False, format='mp3', autoplay=True)#, every=1, value=radio_tick_mp3)
                    radio_update_button = gr.Button("Update", scale=8)
                    radio_update_button.click(agent._radio.update, textboxes, textboxes)
                    radio_play_button = gr.Button("▶️", scale=1)
                    radio_play_button.click(agent._radio.play)
            with gr.Tab('Experimental'):
                gr.Markdown('You are an AI Agent. You are able to reason about text, and also have access to the following tools: * '
                            + '\n* '.join([f"{plugin_name}.{f_name} - {plugin.functions[f_name].description}" for plugin_name, plugin in agent.list_plugins().items() for f_name in plugin.functions])
                            + "\n\nGenerate a list of steps for the following task, where each step specifies the input, the tool or tools to be used, and the expected output."
                            + '\n\nThe task is: Generate a threat model for the system design described at the following URL: <url>')
            with gr.Tab('Memories'):
                docs_trigger = gr.State(1)
                with gr.Row():
                    doc_upload = gr.File(type="filepath", label="Upload a document")
                    doc_upload.upload(agent.process_file_input_memories, doc_upload, doc_upload)
                    gr.Button("Refresh", scale=1).click(lambda x: x + 1, docs_trigger, docs_trigger)
                @gr.render(inputs=docs_trigger)
                def render_documents(docs_trigger):
                    files = agent.documents()
                    with gr.Row():
                        file_box = gr.File(file_count="multiple", type="filepath", value=[x[1] for x in files], interactive=True, key="doc_organiser")
                        file_box.delete(agent.delete_document, [file_box])
                with gr.Row():
                    wiki_uri = gr.Textbox(label="Wiki URI", type="text", placeholder=settings.confluence_uri)
                with gr.Row():
                    cql = gr.Textbox(label="Wiki Search CQL", type="text")
                    wiki_search_button = gr.Button("Search", scale=1)
                    wiki_search_button.click(agent.search_wiki, [cql, chatbot], [chatbot, wav_speaker, mp3_speaker])

demo.queue()

if __name__ == '__main__':
    demo.launch()