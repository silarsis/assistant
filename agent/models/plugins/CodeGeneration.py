from pydantic import BaseModel
from typing import Any, Annotated
import os


from semantic_kernel.functions.kernel_function_decorator import kernel_function

from config import settings


CHARACTER="You are an expert developer who has a special interest in security.\n"
PROMPT="Generate code according to the following specifications:\n{{$input}}"

class CodeGenerationPlugin(BaseModel):
    kernel: Any = None
    prompt: Any = None
    api_key: str = os.environ.get("IMG_OPENAI_API_KEY", settings.openai_api_key)
    base_url: str = os.environ.get("IMG_OPENAI_API_BASE", settings.openai_api_base)
    api_version: str = os.environ.get("IMG_OPENAI_API_VERSION", settings.openai_api_version)
    org_id: str = os.environ.get("IMG_OPENAI_ORG_ID", settings.openai_org_id)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt = self.kernel.create_function_from_prompt(
            function_name="generate_code", plugin_name="code_generation",
            description="Generage code from a specification",
            prompt=CHARACTER + PROMPT, max_tokens=2000, temperature=0.2, top_p=0.5)

    @kernel_function()
    async def gen_code(self, specification: Annotated[str, "The code specification"] = "") -> str:
        " Generage code from a specification "
        ctx = self.kernel.create_new_context()
        ctx.variables["input"] = input
        result = await self.prompt.invoke(context=ctx)
        return str(result).strip()