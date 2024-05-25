import json
import requests
import time
from asyncio import Queue

from pydantic import BaseModel

from config import settings
from models.tools.llm_connect import LLMConnect, AsyncOpenAI

RADIO_STYLE="80's and 90's greatest hits and influential music"
ANNOUNCER_STYLE="Casey Kasem's American Top 40"

RADIO_PROMPT = """
You are a programmer for a radio station that specialises in {radio_style}.
Generate a 5 song playlist and some text for the announcer to say, and an ad to play.
The text for the announcer should be in the style of {announcer_style}.
The ad should be a joke version of the sort of ad you'd find on a radio station with this theme, no more than 15 seconds long.
Never repeat the songs from previous requests. Previous songs already played are:

{history}

Present the playlist and text as json, as follows:

{{
  "tracks": [
    {{
      "name": "With or Without You",
      "artist": "U2"
    }},
    {{
      "name": "Thriller",
      "artist": "Michael Jackson"
    }},
    {{
      "name": "Like a Prayer",
      "artist": "Madonna"
    }},
    {{
      "name": "Losing My Religion",
      "artist": "R.E.M."
    }},
    {{
      "name": "Enter Sandman",
      "artist": "Metallica"
    }}
  ],
  "dj": "We headed up that five with 'With or Without You' by U2 - a song that turned emotional turmoil into a worldwide anthem. Then you heard 'Thriller' by Michael Jackson, a track that not only dominated the charts but also revolutionized the music video industry. Up next was Madonna with 'Like a Prayer', blending pop and gospel in a way that only she could, sparking conversations around the globe. R.E.M. had us 'Losing My Religion', a song that showed us it's okay to question and reflect, becoming an instant classic. And rounding off our set, Metallica's 'Enter Sandman' thundered through, turning nightmares into headbanging anthems. These tracks aren't just songs; they're the soundtrack to our lives, each one a chapter in the story of music. Stay tuned for more chapters as we continue to spin the hits right here."
  "ad": "This hour of programming is brought to you by the new movie 'Top Gun: Maverick'. Catch it in theaters this weekend!"
}}
"""


class AIRadio(BaseModel):
    _llm: AsyncOpenAI = None
    history: str = ''
    radio_style: str = RADIO_STYLE
    announcer_style: str = ANNOUNCER_STYLE
    access_token: str = ''
    refresh_token: str = ''
    expiry: int = 0
    _mp3_queue: Queue = None
    _wav_queue: Queue = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mp3_queue = Queue()
        self._wav_queue = Queue()
        self._llm = LLMConnect(
            api_type=settings.openai_api_type, 
            api_key=settings.openai_api_key, 
            api_base=settings.openai_api_base, 
            deployment_name=settings.openai_deployment_name, 
            org_id=settings.openai_org_id
        ).openai(async_client=True)
        
    def update(self, radio_style, announcer_style):
        self.radio_style = radio_style
        self.announcer_style = announcer_style
        return radio_style, announcer_style
        
    def authorise(self):
        if not settings.spotify_client_code:
            raise PermissionError("Not logged in")
        params = {
            'grant_type': 'authorization_code',
            'code': settings.spotify_client_code,
            'redirect_uri': 'http://localhost:7860/spotify_login',
            'client_id': settings.spotify_client_id,
            'code_verifier': settings.spotify_client_verifier
        }
        auth_response = requests.post("https://accounts.spotify.com/api/token", params=params, headers={'Content-Type': 'application/x-www-form-urlencoded'})
        if auth_response.status_code == 405:
            raise PermissionError("Not logged in")
        self.access_token = auth_response.json()['access_token']
        self.refresh_token = auth_response.json()['refresh_token']
        self.expiry = auth_response.json()['expires_in'] + time.time() - 5 # 5 seconds before for some buffer
        
    def get_spotify_uri(self, song: str, artist: str) -> str:
        headers = {
            'Authorization': f'Bearer {settings.spotify_access_token}'
        }
        params = {
          'q': f'track:{song} artist:{artist}',
          'type': 'track',
          'limit': 1
        }
        response = requests.get('https://api.spotify.com/v1/search', params=params, headers=headers)
        if response.status_code in (400, 401):
            return "Please login to spotify to continue"
        uri = response.json()['tracks']['items'][0]['uri']
        return uri
    
    def play_song(self, uri: str) -> str:
        headers = {
            'Authorization': f'Bearer {settings.spotify_access_token}'
        }
        params = {
          'uri': uri,
        }
        response = requests.get('https://api.spotify.com/v1/me/player/queue', params=params, headers=headers)
        if response.status_code == 403:
            return "Forbidden, maybe you need premium spotify?"
        print(response)
    
    async def play(self, *args, **kwargs) -> str:
        full_prompt = RADIO_PROMPT.format(radio_style=self.radio_style, announcer_style=self.announcer_style, history=self.history)
        playlist = await self._llm.chat.completions.create(messages=[{"role": "system", "content": full_prompt}], model="gpt-4")
        print(playlist)
        try:
            response = json.loads(str(playlist.choices[0].message.content).strip())
        except json.JSONDecodeError:
            return "Failed to generate playlist, json returned was broken"
        for track in response['tracks']:
            self.history += f"{track['name']} by {track['artist']}\n"
            uri = self.get_spotify_uri(track['name'], track['artist'])
            self.play_song(uri)
        # Should send the announcer then the ad for playing here
        # self._mp3_queue.put_nowait(response['dj'])
        # self._mp3_queue.put_nowait(response['ad'])
      
    def get_from_spotify(self, uri: str) -> str:
        headers = {
            'Authorization': f'Bearer {settings.spotify_access_token}'
        }
        return requests.get(uri, headers=headers)