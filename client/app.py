import tkinter as tk
import socket
import threading
import speech_recognition as sr



class Application(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("Echo Client")
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.hostname_label = tk.Label(self, text="Hostname:")
        self.hostname_label.pack()
        self.hostname_entry = tk.Entry(self)
        self.hostname_entry.insert(0, "localhost")
        self.hostname_entry.pack()

        self.port_label = tk.Label(self, text="Port:")
        self.port_label.pack()
        self.port_entry = tk.Entry(self)
        self.port_entry.insert(0, "8000")
        self.port_entry.pack()

        self.connect_button = tk.Button(self, text="Connect", command=self.connect)
        self.connect_button.pack()

        self.disconnect_button = tk.Button(self, text="Disconnect", command=self.disconnect, state=tk.DISABLED)
        self.disconnect_button.pack()

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
        self.response_text = tk.Text(self, height=5, state=tk.DISABLED)
        self.response_text.pack()

        self.socket = None

    def connect(self):
        hostname = self.hostname_entry.get()
        port = int(self.port_entry.get())
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((hostname, port))
        self.connect_button.configure(state=tk.DISABLED)
        self.disconnect_button.configure(state=tk.NORMAL)
        self.send_button.configure(state=tk.NORMAL)
        self.message_entry.configure(state=tk.NORMAL)

    def disconnect(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
            self.connect_button.configure(state=tk.NORMAL)
            self.disconnect_button.configure(state=tk.DISABLED)
            self.send_button.configure(state=tk.DISABLED)
            self.message_entry.configure(state=tk.DISABLED)

    def send(self):
        message = self.message_entry.get()
        self.socket.sendall(message.encode() + b'\n')
        response = self.socket.recv(1024).decode()
        self.response_text.configure(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, response)
        self.response_text.configure(state=tk.DISABLED)
        
    def speech_to_text(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)

        while True:
            with mic as source:
                audio = recognizer.listen(source)
                try:
                    message = recognizer.recognize_google(audio)
                    self.message_entry.delete(0, tk.END)
                    self.message_entry.insert(0, message)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    pass

    def quit(self):
        self.disconnect()
        self.master.destroy()

root = tk.Tk()
app = Application(root)
app.mainloop()
