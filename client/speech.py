import os
import pyttsx3
import requests
from playsound import playsound, PlaysoundException
import queue
import threading
import time
from dotenv import load_dotenv

load_dotenv()

TIMEOUT = 2

class ElevenLabs:
    def __init__(self):
        self._headers = {
            'Content-Type': 'application/json',
            'xi-api-key': os.environ.get('ELEVENLABS_API_KEY')
        }
        self._voices = [os.environ.get('ELEVENLABS_VOICE_1_ID'), os.environ.get('ELEVENLABS_VOICE_2_ID')]
        
    def say(self, text: str):
        print("Requesting TTS", flush=True)
        voice_index = 0
        tts_url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{self._voices[voice_index]}"
        )
        response = requests.post(tts_url, headers=self._headers, json={"text": text})

        if response.status_code == 200:
            with open("speech.mpeg", "wb") as f:
                f.write(response.content)
            try:
                playsound("speech.mpeg", True)
            except PlaysoundException:
                print("Failed to play sound", flush=True)
            os.remove("speech.mpeg")
            return True
        else:
            print("Request failed with status code:", response.status_code)
            print("Response content:", response.content)
            return False
        
class TTS:
    def __init__(self):
        self.engine = pyttsx3.init()
        
    def say(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()
        
class Speech:
    def __init__(self):
        self._ending = False
        if os.environ.get('ELEVENLABS_API_KEY'):
            self.engine = ElevenLabs()
        else:
            self.engine = TTS()
        self._incomingQueue = queue.Queue()
        thread = threading.Thread(target=self.talk)
        thread.start()
        
    def quit(self):
        self._ending = True
        time.sleep(TIMEOUT)
            
    def talk(self):
        " This runs in a separate thread and says words when there's enough to say "
        words_to_say = []
        while True:
            try:
                text = self._incomingQueue.get(block=True, timeout=TIMEOUT)
            except queue.Empty:
                if self._ending:
                    self.engine.say("Shutting down")
                    return
                if words_to_say:  # Assume that any longer delay is a sign of completion
                    self.engine.say(''.join(words_to_say))
                    words_to_say = []
                continue
            else:
                words_to_say.append(text)
                
    def say(self, text: str):
        " This puts incoming words on the queue "
        self._incomingQueue.put(text)