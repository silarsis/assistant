from typing import Any, Optional

from pydantic import BaseModel

from semantic_kernel.plugin_definition import kernel_function
from semantic_kernel.orchestration.kernel_context import KernelContext

import wolframalpha

class WolframAlphaPlugin(BaseModel):
    wolfram_client: Optional[Any] = None
    wolfram_alpha_appid: Optional[str] = None
        
    @kernel_function(
        description="Query Wolfram Alpha for factual or general knowledge questions, math, current events, news headlines, or expert-level answers on topics ranging from science, culture, and history to sports, geography, weather, and more.",
        name="wolfram",
        input_description="The question you want to ask"
    )
    def wolfram(self, query: str) -> str:
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