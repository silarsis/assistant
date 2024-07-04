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
        for p_name, p in self.kernel.plugins.items():
            for f_name, f in p.functions.items():
                functions.append(f"{p_name}.{f_name} - {f.description}")
        str_f = '\n* '.join(functions)
        return str_f