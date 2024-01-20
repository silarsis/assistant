import kivy
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed, ConnectionClosedError
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.clock import mainthread
import google_auth_oauthlib

from talk import Talk
from listen import Listen

import threading
import json

import logging
logging.disable(logging.DEBUG)

kivy.require('2.2.0')

class EchoClient(Widget):
    hostname_entry = ObjectProperty(None)
    port_entry = ObjectProperty(None)
    connect_button = ObjectProperty(None)
    send_button = ObjectProperty(None)
    message_entry = ObjectProperty(None)
    response_text = ObjectProperty(None)
    session_id_entry = ObjectProperty(None)
    hear_thoughts_button = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._credentials = None
        self._ending = False
        self.websocket = None
        self.hear_thoughts = False
        self.talk = Talk() # Setup the talking thread
        self.listen = Listen(self.add_to_message_text) # Setup the listening thread
    
    def connect(self):
        host = self.hostname_entry.text
        port = self.port_entry.text
        uri = f'ws://{host}:{port}'
        try:
            self.websocket = connect(uri)
            print(f"Connected to {uri}")
            self.connect_button.disabled = True
            self.send_button.disabled = False
            thread = threading.Thread(target=self.receive_messages)
            thread.start()
            self.google_login()
        except Exception as e:
            print(f"Failed to connect to {uri}: {e}")
            self.closed_connection()
    
    def receive_messages(self):
        " Websocket receipt loop "
        while True:
            try:
                for message in self.websocket:
                    try:
                        payload = json.loads(message)['payload']
                    except json.decoder.JSONDecodeError:
                        print("Garbled message, ignoring...")
                    self.add_to_response_text(payload)
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
            
    def _actual_send(self, type='prompt', prompt='', command=''):
        json_message = {'type':type, 'prompt':prompt, 'command':command, 'session_id':self.session_id_entry.text}
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

    def send(self):
        message = self.message_entry.text
        if not message:
            print("Nothing to send, not sending")
            return
        self._actual_send(type='prompt', prompt=message)
        self.response_text.text = ''

    def closed_connection(self):
        print("Connection closed")
        if self.websocket:
            self.websocket.close()
            self.websocket = None
        self.connect_button.disabled = False
        self.send_button.disabled = True

    def start_listening(self):
        self.listen.start_listening()
        
    def stop_listening(self):
        self.listen.stop_listening()
        
    def toggle_listening(self):
        if self.listen.listening:
            self.listen_button.text="Start Listening"
            self.listen.stop_listening()
        else:
            self.listen_button.text="Stop Listening"
            self.listen.start_listening()
            
    def toggle_hear_thoughts(self):
        if self.hear_thoughts:
            self.hear_thoughts_button.text="Hear Thoughts"
            self.hear_thoughts = False
        else:
            self.hear_thoughts_button.text="Stop Hearing Thoughts"
            self.hear_thoughts = True

    @mainthread
    def add_to_response_text(self, message) -> None:
        " Called to add text to the response box when it comes back from the AI (in the main thread) "
        self.response_text.text += message
        self.response_text.render()
        
    @mainthread
    def add_to_message_text(self, text: str) -> None:
        " Called to add text to the request box when it's translated from audio (in the listen thread) "
        self.message_entry.text = text
        self.send()
            
    def quit(self):
        print("Quitting")
        self._ending = True
        self.listen.quit()
        self.talk.quit()
        self.closed_connection()
        
    def google_login(self):
        if not self._credentials:
            self._credentials = google_auth_oauthlib.get_user_credentials(
                ["https://www.googleapis.com/auth/drive.readonly", "https://www.googleapis.com/auth/documents.readonly"], 
                # app desktop credentials
                '438635256773-rf4rmv51lo436a576enb74t7pc9n8rre.apps.googleusercontent.com', 
                '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')
        self._actual_send(
            type='system', 
            command='update_google_docs_token', 
            prompt=self._credentials.to_json()
        )

class ClientApp(App):
    def build(self):
        return EchoClient()
    
    def on_stop(self):
        self.root.quit()

if __name__ == '__main__':
    c = ClientApp()
    try:
        c.run()
    except KeyboardInterrupt:
        c.stop()
    except Exception:
        c.stop()
        raise