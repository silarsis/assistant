import os
from playsound import playsound, PlaysoundException
import queue
import threading
from dotenv import load_dotenv
import elevenlabs
import requests
import pyaudio
import wave
from urllib.parse import quote

load_dotenv()

TIMEOUT = 2
CHUNK = 1024

class ElevenLabs:
    def __init__(self):
        elevenlabs.set_api_key(os.environ.get('ELEVENLABS_API_KEY'))
        self._voices = [os.environ.get('ELEVENLABS_VOICE_1_ID'), os.environ.get('ELEVENLABS_VOICE_2_ID')]
        self._stream = True
        
    def _save_and_say(self, audio_stream, filename="speech.mpeg"):
        elevenlabs.save(audio=audio_stream, filename=filename)
        try:
            playsound("speech.mpeg", True)
        except PlaysoundException:
            print("Failed to play sound", flush=True)
        os.remove("speech.mpeg")
        
        
    def say(self, text: str):
        print("Requesting TTS", flush=True)
        try:
            audio_stream = elevenlabs.generate(text=text, voice=self._voices[0], stream=self._stream)
        except elevenlabs.api.error.RateLimitError as err:
            print(str(err), flush=True)
        if self._stream:
            try:
                elevenlabs.stream(audio_stream)
            except ValueError:
                # Raised if mpv can't be found
                print("Disabling streaming (probably) because of lack of mpv")
                self._stream = False
                self.say(text)
        else:
            self._save_and_say(audio_stream)
        return True

class TTS:
    def say(self, text: str):
        q_text = quote(text)
        with requests.get(f"http://localhost:5002/api/tts?text={q_text}&speaker_id=p364&style_wav=&language_id=", stream=True) as wav:
            p = pyaudio.PyAudio()
            
            wf = wave.Wave_read(wav.raw)
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=1,
                            rate=22050,
                            output=True)
            while len(data := wf.readframes(CHUNK)):
                stream.write(data)
            stream.stop_stream()
            stream.close()
            p.terminate()
        
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