import tkinter as tk
from tkinter import ttk
import json
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed, ConnectionClosedError
from talk import Talk
from listen import Listen
import openai
import threading
import os
import time

STOP_STR = "###STOP###"


class Application(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._ending = False
        self.master = master
        self.master.title("Echo Client")
        self.websocket = None
        self.pack()
        self.create_widgets()
        self.talk = Talk() # Setup the talking thread
        self.listen = Listen(self.add_to_message_text) # Setup the listening thread

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

        self.quit_button = tk.Button(self.master, text="Quit", command=self.quit)
        self.quit_button.pack()

        message_frame = tk.LabelFrame(self.master, text="Message:")
        self.message_entry = tk.Text(message_frame, height=5, state=tk.DISABLED)
        self.message_entry.pack(side="left", padx=5, pady=5)
        self.clear_button = tk.Button(message_frame, text="Clear", command=self.clear_message)
        self.clear_button.pack(side="left", padx=5, pady=5)
        self.send_button = tk.Button(message_frame, text="Send", command=self.send, state=tk.DISABLED)
        self.send_button.pack(side="right", padx=5, pady=5)
        message_frame.pack(fill=tk.X)

        response_frame = tk.LabelFrame(self.master, text="Response:")
        self.response_text = tk.Text(response_frame, height=20, wrap=tk.WORD, state=tk.DISABLED)
        self.response_text.pack(side="bottom", fill=tk.BOTH, expand=True)
        response_frame.pack(fill=tk.BOTH)

    def start_listening(self):
        self.listen_button.configure(state=tk.DISABLED)
        self.listen.start_listening()
        self.stop_listening_button.configure(state=tk.NORMAL)
        
    def stop_listening(self):
        self.stop_listening_button.configure(state=tk.DISABLED)
        self.listen.stop_listening()
        self.listen_button.configure(state=tk.NORMAL)
        
    def closed_connection(self):
        print("Connection closed")
        if self.websocket:
            self.websocket.close()
            self.websocket = None
        try:
            self.connect_button.configure(state=tk.NORMAL)
            self.send_button.configure(state=tk.DISABLED)
            self.message_entry.configure(state=tk.DISABLED)
        except:
            pass
    
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
            self.start_listening()
        except Exception as e:
            print(f"Failed to connect to {uri}: {e}")
            self.closed_connection()

    def send(self):
        message = self.message_entry.get(1.0, "end-1c").encode()
        message = message.decode('utf-8')
        if not message:
            print("Nothing to send, not sending")
            return
        try:
            self.websocket.send(json.dumps({'prompt':message, 'type':'prompt'}))
        except (ConnectionClosedError, AttributeError):
            self.closed_connection()
            return
        self.response_text.configure(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.response_text.configure(state=tk.DISABLED)
        
    def add_to_response_text(self, message) -> None:
        " Called to add text to the response box when it comes back from the AI (in the main thread) "
        self.response_text.configure(state=tk.NORMAL)
        self.response_text.insert(tk.END, message)
        self.response_text.configure(state=tk.DISABLED)
        
    def add_to_message_text(self, text: str) -> None:
        " Called to add text to the request box when it's translated from audio (in the listen thread) "
        self.message_entry.after(1, self.message_entry.delete, 1.0, tk.END)
        self.message_entry.after(2, self.message_entry.insert, tk.END, text)
        self.message_entry.after(5, self.send)
        
    def clear_message(self):
        " Only ever called in the main loop as a response to a button press "
        self.message_entry.delete(1.0, tk.END)
        
    def receive_messages(self):
        " Websocket receipt loop "
        while True:
            try:
                for message in self.websocket:
                    try:
                        payload = json.loads(message)['payload']
                    except:
                        print("Garbled message, ignoring...")
                    self.response_text.after(1, self.add_to_response_text, payload)
                    if self.listen.listening:
                        self.talk.say(self.stop_listening, payload, self.start_listening) # Stop listening, say the response, then start listening again
                    else:
                        self.talk.say(lambda: None, payload, lambda: None) # Don't stop listening, just say the response
            except ConnectionClosed:
                return
            except TypeError: # NoneType is not iterable
                return

    def quit(self):
        print("Quitting")
        self._ending = True
        self.listen.quit()
        self.talk.quit()
        self.closed_connection()
        self.master.destroy()

root = tk.Tk()
app = Application(root)
try:
    app.mainloop()
except:
    app.quit()