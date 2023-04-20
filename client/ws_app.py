import tkinter as tk
import asyncio
import websockets
import speech_recognition as sr


class MainWindow:
    def __init__(self, master):
        self.master = master
        master.title("Gabby Client")
        self.create_widgets()
        self.websocket = None

    def create_widgets(self):
        self.textbox = tk.Text(self.master, height=10, width=50)
        self.textbox.pack()

        self.button = tk.Button(self.master, text="Start Listening", command=self.start_listening)
        self.button.pack()

        self.quit_button = tk.Button(self.master, text="Quit", command=self.quit)
        self.quit_button.pack()

    async def connect_websocket(self):
        uri = "ws://localhost:8765"
        while True:
            try:
                self.websocket = await websockets.connect(uri)
                print(f"Connected to {uri}")
                break
            except Exception as e:
                print(f"Failed to connect to {uri}: {e}")
                await asyncio.sleep(1)

    async def listen_for_speech(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            while True:
                print("Listening...")
                audio = r.listen(source)
                try:
                    text = r.recognize_google(audio)
                    print(f"Recognized: {text}")
                    await self.websocket.send(text)
                except Exception as e:
                    print(f"Failed to recognize: {e}")

    async def start_listening(self):
        await self.connect_websocket()
        await asyncio.gather(self.listen_for_speech(), self.receive_messages())

    async def receive_messages(self):
        while True:
            try:
                message = await self.websocket.recv()
                print(f"Received: {message}")
                self.textbox.insert(tk.END, message + "\n")
            except Exception as e:
                print(f"Error receiving message: {e}")
                break

    def quit(self):
        if self.websocket:
            asyncio.get_event_loop().run_until_complete(self.websocket.close())
        self.master.destroy()


root = tk.Tk()
app = MainWindow(root)
root.mainloop()
