from typing import Callable
from models.guide import Guide


class Parser:
    def __init__(self):
        with open('./character.txt', 'r') as char_file:
            self.agent = Guide(default_character=char_file.read())
    
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None], **kwargs):
        print(f"Prompting with {prompt}", flush=True)
        await self.agent.prompt_with_callback(prompt, callback=callback, **kwargs)
        
    def prompt(self, prompt: str, **kwargs) -> str:
        print(f"Prompting with {prompt}", flush=True)
        return self.agent.prompt(prompt, **kwargs)
    
    async def update_google_docs_token(self, token: str, callback: Callable[[str], None], **kwargs) -> str:
        await self.agent.update_google_docs_token(token, callback=callback, **kwargs)