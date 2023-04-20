import asyncio
import sys
import os
from websockets.server import serve
from websockets.exceptions import ConnectionClosedError
import json
from prompt_parser import GPT4All
import nest_asyncio
import uuid
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

if os.getenv('INSECURE') and bool(os.getenv('INSECURE')):
    ssl.SSLContext.verify_mode = property(lambda self: ssl.CERT_NONE, lambda self, newval: None)

class API:
    web_host = '0.0.0.0'
    web_port = 8765
    text_host = '0.0.0.0'
    text_port = 8000
    text_loop = asyncio.new_event_loop()
    ws_loop = asyncio.new_event_loop()
    
    def __init__(self):
        print("Launching bot...", flush=True)
        self.bot = GPT4All()
        
    async def handle_text_connection(self, reader, writer):
        print("Text connection", flush=True)
        while True:
            prompt = await reader.readline()
            if not prompt:
                continue
            callback = lambda x:self.text_loop.run_until_complete(self.text_send_coroutine(x, writer=writer))
            await self.bot.prompt_callback(prompt, callback=callback)
            
    async def text_send_coroutine(self, payload: str, writer):
        writer.write(payload.encode() + b'\n')
        await writer.drain()
            
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
                    callback = lambda x:self.ws_loop.run_until_complete(self.ws_send_coroutine(websocket, m["prompt"], x, 'response', correlation_id))
                    await self.bot.prompt_callback(m["prompt"], callback=callback)
        except ConnectionClosedError:
            print("WS connection broken", flush=True)
            await websocket.close()

    async def main(self):
        print("Launching WS server", flush=True)
        ws_server = await serve(self.handle_ws_connection, self.web_host, self.web_port)
        print(f"WS started on port {self.web_port}", flush=True)
        
        print("Launching text server", flush=True)
        text_server = await asyncio.start_server(self.handle_text_connection, self.text_host, self.text_port)
        print(f'Text started on port {self.text_port}', flush=True)
        
        # async with text_server:
        #     await text_server.serve_forever()
        async with ws_server, text_server:
            await asyncio.gather(text_server.serve_forever(), ws_server.serve_forever())

api = API()
asyncio.run(api.main())