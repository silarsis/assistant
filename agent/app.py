import asyncio
import aiohttp
import os
import ssl
from websockets.server import serve
from websockets.exceptions import ConnectionClosedError
import json
from prompt_parser import Parser
import nest_asyncio
import uuid
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

if os.getenv('INSECURE'):
    print("Disabling SSL checks")
    ssl.SSLContext.verify_mode = property(lambda self: ssl.CERT_NONE, lambda self, newval: None)

class API:
    web_host = '0.0.0.0'
    web_port = 10000
    text_host = '0.0.0.0'
    text_port = 8000
    text_loop = asyncio.new_event_loop()
    ws_loop = asyncio.new_event_loop()
    
    def __init__(self):
        print("Launching bot...")
        self.bot = Parser()
            
    async def text_send_coroutine(self, payload: str, writer):
        writer.write(payload.encode() + b'\n')
        await writer.drain()
        
    async def handle_text_connection(self, reader, writer):
        print("Text connection")
        while True:
            prompt = await reader.readline()
            if not prompt:
                continue
            def callback(x):
                return self.text_loop.run_until_complete(
                    self.text_send_coroutine(x, writer=writer))
            await self.bot.prompt_with_callback(prompt, callback=callback)
            
    async def ws_send_coroutine(self, websocket, prompt: str, payload: str, type: str, correlation_id: str):
        response = {"type": type, "payload": payload, "correlationId": correlation_id, "prompt": prompt}
        await websocket.send(json.dumps(response))

    async def handle_ws_connection(self, websocket):
        print("WS connection")
        try:
            async for message in websocket:
                print("Received message " + str(message))
                try:
                    m=json.loads(message)
                except json.JSONDecodeError as e:
                    print(str(e))
                    continue
                # Check if prompt from web or message from whatsapp
                # {'type': 'prompt', 'prompt': 'prompt text', 'hear_thoughts': True}
                # {'type': 'system', 'command': 'update_prompt_template', 'prompt': 'new prompt'}
                # {'type': 'system', 'command': 'update_character', 'prompt': 'new character'}
                # {'type': 'system', 'command': 'update_google_docs_token', 'prompt': 'new token'}
                cmd = None
                if (m.get("type") == 'prompt'):
                    cmd = self.bot.prompt_with_callback
                elif (m.get("type") == 'system'):
                    if (m.get("command") == 'update_prompt_template'):
                        cmd = self.bot.update_prompt_template
                    if (m.get('command') == 'update_character'):
                        cmd = self.bot.update_character
                    if (m.get('command') == 'update_google_docs_token'):
                        cmd = self.bot.update_google_docs_token
                if cmd:
                    correlation_id=f'{uuid.uuid4()}'
                    def callback(x):
                        return self.ws_loop.run_until_complete(
                            self.ws_send_coroutine(
                                websocket, 
                                m['prompt'], 
                                x, 
                                'response', 
                                correlation_id))
                    await cmd(
                        m["prompt"], 
                        callback=callback, 
                        hear_thoughts=m.get('hear_thoughts', False),
                        session_id=m.get('session_id', 'static'))
        except ConnectionClosedError:
            print("WS connection broken")
            await websocket.close()

    def format_whatsapp_outbound(self, recipient: str, text: str) -> str:
        return json.dumps({
            "messaging_product": "whatsapp",
            "preview_url": False,
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {
                "body": text
            }
        })

    async def send_whatsapp_message(self, text: str):
        # Send a message to whatsapp
        headers = {
            "Content-type": "application/json",
            "Authorization": f"Bearer {os.environ.get('WHATSAPP_ACCESS_TOKEN')}",
        }
        data = self.format_whatsapp_outbound(os.environ.get('WHATSAPP_RECIPIENT_WAID'), text)
        async with aiohttp.ClientSession() as session:
            url = 'https://graph.facebook.com/v13.0/' + f"{os.environ.get('WHATSAPP_PHONE_NUMBER_ID')}/messages"
            try:
                async with session.post(url, data=data, headers=headers) as response:
                    if response.status == 200:
                        print("Status:", response.status)
                        print("Content-type:", response.headers['content-type'])
                        html = await response.text()
                        print("Body:", html)
                    else:
                        print(response.status)        
                        print(response)        
            except aiohttp.ClientConnectorError as e:
                print('Connection Error', str(e))

    async def main(self):
        print("Launching WS server")
        ws_server = await serve(self.handle_ws_connection, self.web_host, self.web_port)
        print(f"WS started on port {self.web_port}")
        
        print("Launching text server")
        text_server = await asyncio.start_server(
            self.handle_text_connection, 
            self.text_host, 
            self.text_port)
        print(f'Text started on port {self.text_port}')
        
        async with ws_server, text_server:
            await asyncio.gather(text_server.serve_forever(), ws_server.serve_forever())

api = API()
asyncio.run(api.main())