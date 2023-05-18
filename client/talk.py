import os
import pyttsx3
import requests
from playsound import playsound, PlaysoundException
import queue
import threading
from dotenv import load_dotenv
import elevenlabs

load_dotenv()

TIMEOUT = 2

class ElevenLabs:
    def __init__(self):
        elevenlabs.set_api_key(os.environ.get('ELEVENLABS_API_KEY'))
        self._headers = {
            'Content-Type': 'application/json',
            'xi-api-key': os.environ.get('ELEVENLABS_API_KEY')
        }
        self._voices = [os.environ.get('ELEVENLABS_VOICE_1_ID'), os.environ.get('ELEVENLABS_VOICE_2_ID')]
        
    def say(self, text: str):
        print("Requesting TTS", flush=True)
        # audio_stream = elevenlabs.generate(text=f'... ... ... {text}', voice=self._voices[0], stream=True)
        # elevenlabs.play(audio_stream)
        voice_index = 0
        tts_url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{self._voices[voice_index]}"
        )
        response = requests.post(tts_url, headers=self._headers, json={"text": f'... ... ... {text}'})

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
        
class Talk:
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
        " Set a flag to stop the talk thread and wait for the timeout - runs in main thread "
        self._ending = True
        #time.sleep(TIMEOUT)
            
    def talk(self):
        " This runs in a separate thread and says words when there's enough to say "
        words_to_say = []
        print("Talk thread started")
        while True:
            try:
                (callback_start, text, callback_stop) = self._incomingQueue.get(block=True, timeout=TIMEOUT)
            except queue.Empty:
                if self._ending:
                    print("Stopping talking")
                    self.engine.say("Shutting down")
                    return
                if words_to_say:  # Assume that any longer delay is a sign of completion
                    callback_start()
                    self.engine.say(''.join(words_to_say))
                    words_to_say = []
                    callback_stop()
                continue
            else:
                words_to_say.append(text)
                
    def say(self, callback_start: callable, text: str, callback_stop: callable):
        " This puts incoming words on the queue, runs in the main thread "
        self._incomingQueue.put((callback_start, text, callback_stop))