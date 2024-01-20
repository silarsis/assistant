import os
import threading
import time
import queue
from dotenv import load_dotenv
import numpy as np
import torch
import speech_recognition as sr
import whisper

if os.environ.get('DEEPGRAM_API_KEY'):
    from deepgram import Deepgram

load_dotenv()

PHRASE_TIMEOUT = 2

# Listener setup stolen from https://github.com/davabase/whisper_real_time/blob/master/transcribe_demo.py

class Listen:
    def __init__(self, callback):
        self.callback = callback
        self._ending = False
        self.listening = False
        self.data_queue = queue.Queue()
        self.bg_stop_fn = None
        self.started_listening = self.stopped_listening = time.time()
        self.model = whisper.load_model("base")
        thread = threading.Thread(target=self.listen)
        thread.start()
        self.setup_background_listener()
        
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
        
    def _bg_listener_callback(self, recognizer, audio):
        " This is called by the background listener when it hears something "
        if not self.listening: # This can accidentally drop some audio, but it's better than nothing
            return
        audio_length = len(audio.get_raw_data()) / audio.sample_width / audio.sample_rate
        print(f"Got audio of length {audio_length} seconds")
        if time.time() - audio_length < self.started_listening:
            return
        self.data_queue.put(audio.get_raw_data())
        
    def setup_background_listener(self):
        recorder = sr.Recognizer()
        recorder.energy_threshold = 1000
        recorder.dynamic_energy_threshold = False
        record_timeout = 2
        source = sr.Microphone(sample_rate=16000)
        # with sr.Microphone(sample_rate=16000) as source:
        with source: # Done this way because it needs to be in a contact to be adjusted
            recorder.adjust_for_ambient_noise(source)
        self.bg_stop_fn = recorder.listen_in_background(source, 
            self._bg_listener_callback, 
            phrase_time_limit=record_timeout)
            
    def slurp_data(self) -> str:
        items = []
        text = ''
        while not self.data_queue.empty():
            try:
                items.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        if items and self.listening:
            print(f"Slurping {len(items)} items")
            # Convert in-ram buffer to something the model can use directly without needing a temp file.
            # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
            # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
            audio_data = b''.join(items)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            result = self.model.transcribe(audio_np, fp16=torch.cuda.is_available())
            text = result['text'].strip()
            print(f"Transcribed: {text}")
        return text
        
    def listen(self):
        " This runs in a separate thread and calls speech-to-text when it hears something "
        last_heard = time.time()
        transcription = []
        while not self._ending:
            time.sleep(0.25)
            text = self.slurp_data()
            if text:
                transcription.append(text)
                last_heard = time.time()
            if transcription and (time.time() - last_heard > PHRASE_TIMEOUT):
                self.callback(' '.join(transcription).strip())
                transcription = []
        print("Stopping listening")
        if self.bg_stop_fn:
            self.bg_stop_fn(wait_for_stop=False)

            