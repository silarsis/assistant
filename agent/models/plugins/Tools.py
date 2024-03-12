from pydantic import BaseModel
from typing import Any
from semantic_kernel.functions.kernel_function_decorator import kernel_function


class ToolsPlugin(BaseModel):
    kernel: Any = None
    
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        
    @kernel_function(description="Returns a list of all the tools the agents have available to them", name="list_tools")
    async def list_tools(self) -> str:
        """Returns a list of all the tools the agents have available to them"""
        functions = []
        for p in self.kernel.plugins:
            for f in p.functions:
                functions.append(f"{p.name}.{f} - {p.functions[f].description}")
        str_f = '\n* '.join(functions)
        return str_f