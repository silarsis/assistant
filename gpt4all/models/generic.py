import abc
from typing import Callable


class ModelClass(abc.ABC):
    @abc.abstractmethod
    def __init__(self, name: str, prompt_template: str):
        # Setup the model and any supporting data structures
        pass
    
    @abc.abstractmethod
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None]):
        # Write a prompt to the bot and callback with the response.
        pass
    
    @abc.abstractmethod
    def prompt(self, prompt: str):
        # Write a prompt to the bot and return the response.
        pass