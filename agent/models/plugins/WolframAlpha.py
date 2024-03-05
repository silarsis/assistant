from typing import Any, Optional, Annotated

from pydantic import BaseModel

from semantic_kernel.functions.kernel_function_decorator import kernel_function

import wolframalpha

class WolframAlphaPlugin(BaseModel):
    wolfram_client: Optional[Any] = None
    wolfram_alpha_appid: Optional[str] = None
        
    @kernel_function()
    def wolfram(self, query: Annotated[str, "The query to send to Wolfram Alpha"]) -> str:
        " Query Wolfram Alpha "
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