from pathlib import Path
from huggingface_hub import hf_hub_download
from pyllamacpp.model import Model

class GPT4All:
    def __init__(self):
        # Make sure we have the model we want to use
        print("Downloading / ensuring model exists...", flush=True)
        hf_hub_download(repo_id="LLukas22/gpt4all-lora-quantized-ggjt", filename="ggjt-model.bin", local_dir=".")
        print("Model download complete", flush=True)
        # Load the model
        self.model = Model(ggml_model='ggjt-model.bin', n_ctx=128)
        print("Model loaded", flush=True)
        
    async def prompt_callback(self, prompt, callback):
        """
        Write a prompt to the bot and return the response.
        """
        print(f"Prompting with {prompt}", flush=True)
        self.model.generate(prompt, n_predict=512, new_text_callback=callback, n_threads=8)
        print("Prompt all done", flush=True)
        