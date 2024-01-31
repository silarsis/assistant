from typing import Callable
from models.guide import Guide
import os


class Parser:
    def __init__(self):
        # Checking in case it's launched from the top dir rather than
        # the agent dir
        if os.path.exists('./character.txt'):
            filename = './character.txt'
        elif os.path.exists('./agent/character.txt'):
            filename = './agent/character.txt'
        with open(filename, 'r') as char_file:
            self.agent = Guide(default_character=char_file.read())
    
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None], **kwargs):
        print(f"Prompting with {prompt}", flush=True)
        return await self.agent.prompt_with_callback(prompt, callback=callback, **kwargs)
    
    def update_google_docs_token(self, token: str, callback: Callable[[str], None], **kwargs) -> str:
        self.agent.update_google_docs_token(token, callback=callback, **kwargs)