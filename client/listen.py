import os
import threading
import time
from dotenv import load_dotenv
import speech_recognition as sr
import whisper

if os.environ.get('DEEPGRAM_API_KEY'):
    from deepgram import Deepgram

load_dotenv()

TIMEOUT = 5

class Listen:
    def __init__(self, callback):
        self.callback = callback
        self._ending = False
        self.listening = False
        self.started_listening = self.stopped_listening = time.time()
        self.model = whisper.load_model("base")
        thread = threading.Thread(target=self.listen)
        thread.start()
        
    def quit(self):
        " Set a flag to stop the talk thread and wait for the timeout - runs in main thread "
        self._ending = True
        #time.sleep(TIMEOUT)
        
    def start_listening(self):
        print("Start listening")
        self.started_listening = time.time()
        self.listening = True
        
    def stop_listening(self):
        print("Stop listening")
        self.listening = False
        self.stopped_listening = time.time()
            
    def listen(self):
        " This runs in a separate thread and calls speech-to-text when it hears something "
        r = sr.Recognizer()
        with sr.Microphone() as source:
            while not self._ending:
                audio = None
                if not self.listening:
                    time.sleep(0.5)
                    continue
                start = time.time()
                try:
                    audio = r.listen(source, timeout=TIMEOUT)
                except sr.WaitTimeoutError:
                    continue
                if time.time() - start < 3:
                    continue
                if not self.listening or self.stopped_listening > start:
                    continue
                if not audio:
                    continue
                try:
                    print("Speech-to-textifying")
                    if os.environ.get('DEEPGRAM_API_KEY'):
                        deepgram = Deepgram(os.environ.get('DEEPGRAM_API_KEY'))
                        audio_data = {'buffer': audio.get_wav_data(), 'mimetype': 'audio/wav'}
                        response = deepgram.transcription.sync_prerecorded(audio_data, {'punctuate': True})
                        print(response)
                        text = response['results']['channels'][0]['alternatives'][0]['transcript']
                    else:
                        audio = whisper.pad_or_trim(audio)
                        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
                        _, probs = self.model.detect_language(mel)
                        print(f"Detected language: {max(probs, key=probs.get)}")
                        options = whisper.DecodingOptions()
                        result = whisper.decode(self.model, mel, options)
                        text = result.text
                    text = str(text)
                    print(f"Recognized: {text}")
                    self.callback(text)
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print("Could not request results from Speech Recognition service; {0}".format(e))
                except ConnectionResetError:
                    print("Connection reset by Speech Recognition service")
            print("Stopped Listening")