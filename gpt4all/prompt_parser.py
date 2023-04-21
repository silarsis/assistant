from pathlib import Path
from huggingface_hub import hf_hub_download
from pyllamacpp.model import Model

class GPT4All:
    def __init__(self):
        self._message_history = []
        # Make sure we have the model we want to use
        print("Downloading / ensuring model exists...", flush=True)
        hf_hub_download(
            repo_id="LLukas22/gpt4all-lora-quantized-ggjt", 
            filename="ggjt-model.bin", 
            local_dir=".")
        print("Model download complete", flush=True)
        # Load the model
        self.model = Model(ggml_model='ggjt-model.bin', n_ctx=2000)
        print("Model loaded", flush=True)
        
    async def prompt_callback(self, prompt: str, callback):
        """
        Write a prompt to the bot and return the response.
        """
        print(f"Prompting with {prompt}", flush=True)
        self._message_history.append({'user': prompt})
        def cb(text):
            self.callback_wrapper(text, callback)
        self.model.generate(
            self.wrapped_prompt(prompt), 
            n_predict=512, 
            new_text_callback=cb, 
            n_threads=8)
        print("Prompt all done", flush=True)
        
    def callback_wrapper(self, text: str, callback):
        " Record the response after stripping out the initial prompt "
        current_item = self._message_history[-1]
        # Record all past plus this token
        response_so_far = current_item.setdefault('bot', '')
        bot  = response_so_far + text
        self._message_history[-1]['bot'] = bot
        # Only respond once we've finished with the prompt
        if len(bot) > len(current_item['full_prompt']):
            callback(text)
        
    def _translate(self, history: dict[str, str]):
        user = history.get('user', '')
        bot = history.get('bot', '').split('\n')[-1]
        return f'User: {user}\nZen: {bot}\n'
        
    def wrapped_prompt(self, prompt: str):
        " Take user input, wrap it in a conversational prompt, and return the result "
        history_of_questions = '\n'.join(
            [ self._translate(item) for item in self._message_history ])
        full_prompt = f"""
You are "Zen", an AI assistant modelled after the Zen character from Blake's 7.
You are designed to be helpful, but you are also a bit of a smartass.
Your goal is to provide the user with answers to their questions and sometimes make them laugh.
You will only provide the "Zen" portion of the below exchanges.

User: Hi there, who are you?
Zen: I am Zen, the ship's computer.
{history_of_questions}
User: {prompt}
        """
        self._message_history[-1]['full_prompt'] = full_prompt
        return full_prompt