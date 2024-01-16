from typing import Any, Optional

from pydantic import BaseModel

from semantic_kernel.skill_definition import sk_function
from semantic_kernel.orchestration.sk_context import SKContext

import wolframalpha

class WolframAlphaPlugin(BaseModel):
    wolfram_client: Optional[Any] = None
    wolfram_alpha_appid: Optional[str] = None
        
    @sk_function(
        description="Query WolframAlpha for factual questions about math, science, society, the time or culture",
        name="wolfram",
        input_description="The question you want to ask"
    )
    def wolfram(self, query: str, context: SKContext) -> str:
        if not self.wolfram_client:
            self.wolfram_client = wolframalpha.Client(self.wolfram_alpha_appid)
        res = self.wolfram_client.query(query)
        try:
            assumption = next(res.pods).text
            answer = next(res.results).text
        except StopIteration:
            return "Wolfram Alpha wasn't able to answer it"
        if not answer:
            # We don't want to return the assumption alone if answer is empty
            return "No good Wolfram Alpha Result was found"
        else:
            return f"Assumption: {assumption} \nAnswer: {answer}"