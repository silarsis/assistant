from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from .generic import ModelClass
from typing import Callable
import torch


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
# - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
# - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
# - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
# - StableLM will refuse to participate in anything that could harm a human.
# """

# prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"

class StableLM(ModelClass):
    def __init__(self, name: str, prompt_template: str):
        self._message_history = []
        self.name = name
        self._prompt_template = '<|SYSTEM|>' + prompt_template + '{history_of_questions}\n'
        self.tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-tuned-alpha-7b")
        print("Loaded Tokenizer", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            "StabilityAI/stablelm-tuned-alpha-7b", 
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16)
        print("Loaded Model", flush=True)
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
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]))
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)
    
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None]) -> None:  # noqa: E501
        prompt = self._wrapped_prompt(prompt)
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        def cb(text):
            self._callback_wrapper(text, callback)
        self.model.generate(
            **inputs,
            new_text_callback=cb,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]))
        print("Prompt all done", flush=True)

    def _callback_wrapper(self, text: str, callback: Callable[[str], None]):
        " Record the response after stripping out the initial prompt "
        callback(self.tokenizer.decode(text, skip_special_tokens=True))
        
    def _translate(self, history: dict[str, str]):
        user = history.get('user', '')
        bot = history.get('bot', '').split('\n')[-1]
        if bot:
            return f'<|USER|>{user}\n<|ASSISTANT|>{bot}'
        return f'<|USER|>{user}'
        
    def _wrapped_prompt(self, prompt: str):
        " Take user input, wrap it in a conversational prompt, and return the result "
        history_of_questions = '\n\n'.join(
            [ self._translate(item) for item in self._message_history ][-20:]) # Note, this includes the current prompt
        full_prompt = self._prompt_template.format(history_of_questions=history_of_questions, name=self.name) + '<|ASSISTANT|>'
        self._message_history[-1]['full_prompt'] = full_prompt
        return full_prompt