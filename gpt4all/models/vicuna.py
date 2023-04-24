from transformers import AutoTokenizer, AutoModelForCausalLM
from .generic import ModelClass
from typing import Callable

class Model(ModelClass):
    def __init__(self, name: str, prompt_template: str):
        self._message_history = []
        self.name = name
        self._prompt_template = prompt_template + '{history_of_questions}\n'
        self.tokenizer = AutoTokenizer.from_pretrained("CarperAI/vicuna-13b-fine-tuned-rlhf")
        self.model = AutoModelForCausalLM.from_pretrained("CarperAI/vicuna-13b-fine-tuned-rlhf")
        self.model.half().cuda()
        print("Model loaded", flush=True)
        
    def update_prompt_template(self, new_template: str = ''):
        self._prompt_template = new_template + '{history_of_questions}\n'
        
    def prompt(self, prompt: str):
        """
        Write a prompt to the bot and return the response.
        """
        prompt = self._wrapped_prompt(prompt)
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=1.0,
            top_p=1.0)
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)
    
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None]) -> None:  # noqa: E501
        prompt = self._wrapped_prompt(prompt)
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        def cb(text):
            self._callback_wrapper(text, callback)
        self.model.generate(
            **inputs,
            new_text_callback=cb,
            max_new_tokens=256,
            do_sample=True,
            temperature=1.0,
            top_p=1.0)
        print("Prompt all done", flush=True)

    def _callback_wrapper(self, text: str, callback: Callable[[str], None]):
        " Record the response after stripping out the initial prompt "
        callback(self.tokenizer.decode(text, skip_special_tokens=True))
        
    def _wrapped_prompt(self, prompt: str):
        " Take user input, wrap it in a conversational prompt, and return the result "
        history_of_questions = '\n\n'.join(
            [ self._translate(item) for item in self._message_history ][-20:]) # Note, this includes the current prompt
        full_prompt = self._prompt_template.format(history_of_questions=history_of_questions, name=self.name)
        self._message_history[-1]['full_prompt'] = full_prompt
        return full_prompt