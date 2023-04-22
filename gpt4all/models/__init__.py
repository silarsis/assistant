from .gpt4all import GPT4All
from .vicuna import Vicuna
from .stablelm import StableLM

def get(model_name: str, name: str, prompt_template: str):
    if model_name == 'vicuna':
        return Vicuna(name, prompt_template)
    elif model_name == 'gpt4all':
        return GPT4All(name, prompt_template)
    elif model_name == 'stablelm':
        return StableLM(name, prompt_template)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    