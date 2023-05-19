import tkinter as tk
import json
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed, ConnectionClosedError
from talk import Talk
from listen import Listen
import threading


class Application(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._ending = False
        self.master = master
        self.master.title("Echo Client")
        self.websocket = None
        self.hear_thoughts = False
        self.session_id = 'client'
        self.pack()
        self.create_widgets()
        self.talk = Talk() # Setup the talking thread
        self.listen = Listen(self.add_to_message_text) # Setup the listening thread

    def create_widgets(self):
        # Host to connect to
        host_frame = tk.LabelFrame(self.master, text="Host")
        self.connected = False
        self.hostname_label = tk.Label(host_frame, text="Hostname:")
        self.hostname_label.pack(side="left", padx=5, pady=5)
        self.hostname_entry = tk.Entry(host_frame)
        self.hostname_entry.insert(0, "localhost")
        self.hostname_entry.pack(side="left", padx=5, pady=5)
        self.port_label = tk.Label(host_frame, text="Port:")
        self.port_label.pack(side="left", padx=5, pady=5)
        self.port_entry = tk.Entry(host_frame)
        self.port_entry.insert(0, "10000")
        self.port_entry.pack(side="left", padx=5, pady=5)
        self.connect_button = tk.Button(host_frame, text="Connect", command=self.connect)
        self.connect_button.pack(side="left", padx=5, pady=5)
        host_frame.pack(fill=tk.X)
        
        # Other controls
        control_frame = tk.LabelFrame(self.master, text="Controls")
        # Listening toggle
        listen_frame = tk.Frame(control_frame)
        self.listening = False
        self.listen_button = tk.Button(listen_frame, text="Start Listening", command=self.toggle_listening)
        self.listen_button.pack(padx=5, pady=5)
        listen_frame.pack(fill=tk.Y, side="left", padx=5, pady=5)
        # Variables to control the response
        var_frame = tk.Frame(control_frame)
        self.hear_thoughts_button = tk.Button(var_frame, text="Hear Thoughts", command=self.toggle_hear_thoughts)
        self.hear_thoughts_button.pack(side="left", padx=5, pady=5)
        self.session_id_label = tk.Label(var_frame, text="Session ID:")
        self.session_id_entry = tk.Entry(var_frame)
        self.session_id_entry.insert(0, self.session_id)
        self.session_id_label.pack(side="left", padx=5, pady=5)
        self.session_id_entry.pack(side="left", padx=5, pady=5)
        var_frame.pack(fill=tk.Y, side="left", padx=5, pady=5)
        # Quit button
        self.quit_button = tk.Button(control_frame, text="Quit", command=self.quit)
        self.quit_button.pack(side="right", padx=5, pady=5)
        control_frame.pack(fill=tk.X)

        message_frame = tk.LabelFrame(self.master, text="Message")
        self.message_entry = tk.Text(message_frame, height=5, state=tk.DISABLED)
        self.message_entry.pack(side="left", padx=5, pady=5)
        self.clear_button = tk.Button(message_frame, text="Clear", command=self.clear_message)
        self.clear_button.pack(side="left", padx=5, pady=5)
        self.send_button = tk.Button(message_frame, text="Send", command=self.send, state=tk.DISABLED)
        self.send_button.pack(side="right", padx=5, pady=5)
        message_frame.pack(fill=tk.X)

        response_frame = tk.LabelFrame(self.master, text="Response")
        self.response_text = tk.Text(response_frame, height=20, wrap=tk.WORD, state=tk.DISABLED)
        self.response_text.pack(side="bottom", fill=tk.BOTH, expand=True)
        response_frame.pack(fill=tk.BOTH)

    def start_listening(self):
        self.listen.start_listening()
        
    def stop_listening(self):
        self.listen.stop_listening()
        
    def toggle_listening(self):
        if self.listening:
            self.listen_button.configure(text="Start Listening")
            self.listening = False
            self.listen.stop_listening()
        else:
            self.listen_button.configure(text="Stop Listening")
            self.listening = True
            self.listen.start_listening()
        
    def toggle_hear_thoughts(self):
        if self.hear_thoughts:
            self.hear_thoughts_button.configure(text="Hear Thoughts")
            self.hear_thoughts = False
        else:
            self.hear_thoughts_button.configure(text="Stop Hearing Thoughts")
            self.hear_thoughts = True
        
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
        except Exception as e:
            print(f"Failed to connect to {uri}: {e}")
            self.closed_connection()

    def send(self):
        message = self.message_entry.get(1.0, "end-1c").encode()
        message = message.decode('utf-8')
        if not message:
            print("Nothing to send, not sending")
            return
        session_id = self.session_id_entry.get()
        if not session_id:
            session_id = 'client'
        json_message = {'prompt':message, 'type':'prompt', 'session_id':session_id}
        if self.hear_thoughts:
            json_message['hear_thoughts'] = True
        attempts = 0
        while attempts < 3:
            try:
                self.websocket.send(json.dumps(json_message))
                break
            except (ConnectionClosedError, AttributeError):
                self.connect()
        else:
            # Executed if we don't break out of the loop
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
                        self.talk.say(self.toggle_listening, payload, self.toggle_listening) # Stop listening, say the response, then start listening again
                    else:
                        self.talk.say(lambda: None, payload, lambda: None) # Don't stop listening, just say the response
            except ConnectionClosed:
                self.closed_connection()
                return
            except TypeError: # NoneType is not iterable
                self.closed_connection()
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