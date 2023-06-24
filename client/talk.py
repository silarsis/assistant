import os
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
        audio_stream = elevenlabs.generate(text=f'... ... ... {text}', voice=self._voices[0], stream=self._stream)
        # elevenlabs.play(audio_stream)
        if self._stream:
            try:
                elevenlabs.stream(audio_stream)
            except ValueError:
                # Raised if mpv can't be found
                self._stream = False
                self.say(text)
        else:
            self._save_and_say(audio_stream)
        return True

        
class TTS:
    def __init__(self):
        print("No supported text to speech currently, setup an elevenlabs account and set ELEVENLABS_API_KEY in .env")
        # self.engine = pyttsx3.init() # This segfaults on Mac python > 3.6.15
        self.engine = None
        
    def say(self, text: str):
        return
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