import tkinter as tk
import json
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed, ConnectionClosedError
import speech_recognition as sr
import threading


class Application(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("Zen Client")
        self.websocket = None
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # Host to connect to
        self.connected = False
        self.hostname_label = tk.Label(self, text="Hostname:")
        self.hostname_label.pack()
        self.hostname_entry = tk.Entry(self)
        self.hostname_entry.insert(0, "localhost")
        self.hostname_entry.pack()
        self.port_label = tk.Label(self, text="Port:")
        self.port_label.pack()
        self.port_entry = tk.Entry(self)
        self.port_entry.insert(0, "8765")
        self.port_entry.pack()
        self.connect_button = tk.Button(self, text="Connect", command=self.connect)
        self.connect_button.pack()
        
        # Listening toggle
        self.listening = False
        self.listen_button = tk.Button(self.master, text="Start Listening", command=self.start_listening)
        self.listen_button.pack()
        self.stop_listening_button = tk.Button(self.master, text="Stop Listening", command=self.stop_listening, state=tk.DISABLED)
        self.stop_listening_button.pack()

        self.send_button = tk.Button(self, text="Send", command=self.send, state=tk.DISABLED)
        self.send_button.pack()

        self.quit_button = tk.Button(self, text="Quit", command=self.quit)
        self.quit_button.pack()

        self.message_label = tk.Label(self, text="Message:")
        self.message_label.pack()
        self.message_entry = tk.Entry(self, state=tk.DISABLED)
        self.message_entry.pack()

        self.response_label = tk.Label(self, text="Response:")
        self.response_label.pack()
        self.response_text = tk.Text(self, height=20, state=tk.DISABLED)
        self.response_text.pack(fill=tk.BOTH, expand=True)

    def start_listening(self):
        self.listen_button.configure(state=tk.DISABLED)
        self.listening = True
        self.stop_listening_button.configure(state=tk.NORMAL)
        thread = threading.Thread(target=self.listen_for_speech)
        thread.start()
        
    def stop_listening(self):
        self.stop_listening_button.configure(state=tk.DISABLED)
        self.listening = False
        self.listen_button.configure(state=tk.NORMAL)
        
    def closed_connection(self):
        print("Connection closed")
        if self.websocket:
            self.websocket.close()
            self.websocket = None
        self.connect_button.configure(state=tk.NORMAL)
        self.send_button.configure(state=tk.DISABLED)
        self.message_entry.configure(state=tk.DISABLED)
    
    def connect(self):
        host = self.hostname_entry.get()
        port = self.port_entry.get()
        uri = f'ws://{host}:{port}'
        try:
            self.websocket = connect(uri)
            print(f"Connected to {uri}")
            self.connect_button.configure(state=tk.DISABLED)
            self.send_button.configure(state=tk.NORMAL)
            self.message_entry.configure(state=tk.NORMAL)
            thread = threading.Thread(target=self.receive_messages)
            thread.start()
        except Exception as e:
            print(f"Failed to connect to {uri}: {e}")
            self.closed_connection()

    def send(self):
        message = self.message_entry.get().encode()
        try:
            self.websocket.send(json.dumps({'prompt':message.decode('utf-8'), 'type':'prompt'}))
        except ConnectionClosedError:
            self.closed_connection()
            return
        self.response_text.configure(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.response_text.configure(state=tk.DISABLED)
        
    def add_to_response_text(self, message):
        self.response_text.configure(state=tk.NORMAL)
        self.response_text.insert(tk.END, message)
        self.response_text.configure(state=tk.DISABLED)
        
    def receive_messages(self):
        while True:
            try:
                for message in self.websocket:
                    print(f"Received: {message}")
                    try:
                        payload = json.loads(message)['payload']
                    except:
                        print("Garbled message, ignoring...")
                    self.response_text.after(10, self.add_to_response_text, payload)
            except ConnectionClosed:
                return
            except TypeError: # NoneType is not iterable
                return

    def listen_for_speech(self):
        if not self.listening:
            return
        r = sr.Recognizer()
        with sr.Microphone() as source:
            while self.listening:
                print("Listening...")
                try:
                    audio = r.listen(source, timeout=5)
                except sr.WaitTimeoutError:
                    continue
                try:
                    text = r.recognize_google(audio)
                    text = str(text)
                    print(f"Recognized: {text}")
                    self.message_entry.after(10, self.message_entry.insert, tk.END, text)
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))

    def quit(self):
        self.closed_connection()
        self.master.destroy()
        
root = tk.Tk()
app = Application(root)
app.mainloop()
