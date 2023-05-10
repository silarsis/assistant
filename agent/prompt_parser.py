from typing import Callable
from models.agent import Agent


class Parser:
    def __init__(self):
        with open('./character.txt', 'r') as char_file:
            self.agent = Agent(character=char_file.read())
    
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None]):
        print(f"Prompting with {prompt}", flush=True)
        await self.agent.prompt_with_callback(prompt, callback=callback)
        
    def prompt(self, prompt: str) -> str:
        print(f"Prompting with {prompt}", flush=True)
        return self.agent.prompt(prompt)