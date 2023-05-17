import asyncio
import os
import ssl
from websockets.server import serve
from websockets.exceptions import ConnectionClosedError
import models
import json
import nest_asyncio
import uuid
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

if os.getenv('INSECURE'):
    print("Disabling SSL checks", flush=True)
    ssl.SSLContext.verify_mode = property(lambda self: ssl.CERT_NONE, lambda self, newval: None)

class API:
    web_host = '0.0.0.0'
    web_port = 8766
    text_loop = asyncio.new_event_loop()
    ws_loop = asyncio.new_event_loop()
    
    def __init__(self):
        print("Launching bot...", flush=True)
        self.bot = models.get(os.environ.get('MODEL', 'vicuna'))
            
    async def ws_send_coroutine(self, websocket, prompt: str, payload: str, type: str, correlation_id: str):
        response = {"type": type, "payload": payload, "correlationId": correlation_id, "prompt": prompt}
        await websocket.send(json.dumps(response))

    async def handle_ws_connection(self, websocket):
        print("WS connection", flush=True)
        try:
            async for message in websocket:
                print("Received message " + str(message), flush=True)
                try:
                    m=json.loads(message)
                except json.JSONDecodeError as e:
                    print(str(e), flush=True)
                    continue
                if (m.get("type") == 'prompt'):
                    correlation_id=f'{uuid.uuid4()}'
                    def callback(x):
                        return self.ws_loop.run_until_complete(
                            self.ws_send_coroutine(
                                websocket, 
                                m['prompt'], 
                                x, 
                                'response', 
                                correlation_id))
                    await self.bot.prompt_with_callback(m["prompt"], callback=callback)
                # {'type': 'system', 'command': 'update_prompt_template', 'prompt': 'new prompt'}
                if (m.get("type") == 'system'):
                    if (m.get("command") == 'update_prompt_template'):
                        self.bot.update_prompt_template(m.get("prompt"))
        except ConnectionClosedError:
            print("WS connection broken", flush=True)
            await websocket.close()

    async def main(self):
        print("Launching WS server", flush=True)
        ws_server = await serve(self.handle_ws_connection, self.web_host, self.web_port)
        print(f"WS started on port {self.web_port}", flush=True)
        
        await ws_server.serve_forever()

api = API()
asyncio.run(api.main())