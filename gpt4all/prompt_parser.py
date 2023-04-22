from typing import Callable
from models.gpt4all import GPT4All


class Parser:
    def __init__(self):
        self.update_prompt_template()
        # Hard coded to GTP4All atm
        self._model = GPT4All('Echo', self._prompt_template)
    
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None]):
        print(f"Prompting with {prompt}", flush=True)
        await self._model.prompt_with_callback(prompt, callback=callback)
        
    def prompt(self, prompt: str) -> str:
        print(f"Prompting with {prompt}", flush=True)
        return self._model.prompt(prompt)
    
    def update_prompt_template(self, new_template: str = ''):
        if new_template:
            self._prompt_template = new_template
        else:
            self._prompt_template = """
Your name is Echo and you are an AI assistant
You are designed to be helpful, but you are also a bit of a smartass.
Your goal is to provide the user with answers to their questions and sometimes make them laugh.
You are to provide Echo's responses to the conversation, using the below exchanges as an example and as history of the conversation so far.
You are not to provide the User's responses. Only provide one response per request.

User: Hi there, what's your name?
Echo: My name is Echo, what's yours?

User: What is your purpose?
Echo: To entertain and inform.

"""