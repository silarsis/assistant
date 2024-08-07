from typing import Optional, List, Tuple, TypedDict, Literal, Required
import os
import json

import semantic_kernel as sk
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable


class Message(TypedDict):
    role: Literal["Human", "AI"]
    content: Required[str]

class SummariseConversation:
    "Designed to summarise a conversation history"

    prompt = """
Summarise the following conversation history, taking the existing context into account:

Context:
{{$context}}

History:
{{$history}}

Summary: """

    def __init__(self, kernel: sk.Kernel, service_id: str = '', character: str = ""):
        self.kernel = kernel
        req_settings = kernel.get_service(service_id).get_prompt_execution_settings_class()(service_id=service_id)
        req_settings.max_tokens = 2000
        req_settings.temperature = 0.2
        req_settings.top_p = 0.5
        self.prompt_template_config = PromptTemplateConfig(
            template=self.prompt, 
            name="summarise_conversation", 
            input_variables=[
                InputVariable(name="context", description="The context of the conversation", required=True),
                InputVariable(name="history", description="The chat history", required=True),
            ],
            execution_settings=req_settings
        )
        self.chat_fn = self.kernel.add_function(
            function_name="summarise_conversation", 
            plugin_name="memory",
            description="Summarise a conversation for an ongoing rolling memory, only used by the memory plugin",
            prompt_template_config=self.prompt_template_config
        )

    async def response(self, context: str="", history: str="") -> str:
        result = await self.kernel.invoke(self.chat_fn, context=context, history=history)
        return str(result).strip()
    
    
class LocalMemory:
    context: Optional[str] = None
    template: str = """
Summarise the following conversation history, taking the existing context into account:

Context:
{{$context}}

History:
{{$history}}

Summary: 
"""
    
    def __init__(self, kernel: sk.Kernel=None, service_id: str=''):
        print("Local Memory")
        self.context = {}
        self.messages = {}
        self.kernel = kernel
        self.summariser = SummariseConversation(kernel, service_id=service_id)
        
    def refresh_from(self, session_id: str) -> None:
        self.load(session_id)
        self.context.setdefault(session_id, "")
        self.messages.setdefault(session_id, [])
        
    def save(self, session_id: str) -> None:
        # Save messages to a file, fix the error if the dir doesn't exist
        os.makedirs(".data", exist_ok=True)
        data = {
            'context': self.context.setdefault(session_id, ""),
            'messages': self.messages.get(session_id, [])
        }
        with open(f".data/{session_id}.json", "w") as f:
            f.write(json.dumps(data))
                
    def load(self, session_id: str) -> None:
        # Load messages from file
        try:
            with open(f".data/{session_id}.json", "r") as f:
                data = json.loads(f.read())
                self.context[session_id] = data['context']
                self.messages[session_id] = [Message(**m) for m in data['messages']]
        except FileNotFoundError:
            print("No memory, starting from scratch")
            pass
        except json.decoder.JSONDecodeError as e:
            print(f"Failed to decode memory: {e}")
            pass
    
    async def _summarise(self, session_id: str) -> str:
        contextualise = self.messages[session_id][:-10]
        if not contextualise:
            return
        self.messages[session_id] = self.messages[session_id][-10:]
        response = await self.summariser.response(context=self.context.setdefault(session_id, ""), history="\n".join([f"{message['role']}: {message['content']}" for message in contextualise]))
        self.context[session_id] = response
        return response
    
    async def add_message(self, role: str, content: str, session_id: str) -> None:
        self.messages.setdefault(session_id, []).append(Message(role=role, content=content))
        if len(self.messages[session_id]) > 20:
            await self._summarise(session_id)
        self.save(session_id)
        
    def get_history(self, session_id: str) -> List[str]:
        if session_id not in self.messages:
            self.refresh_from(session_id)
        return [message["content"] for message in self.messages.setdefault(session_id, [])]

    def get_formatted_history(self, session_id: str) -> str:
        history = self.get_history(session_id)
        return "\n".join(history)
    
    def get_history_for_chatbot(self, session_id: str) -> List[Tuple[str, str]]:
        if session_id not in self.messages:
            self.refresh_from(session_id)
        return self.messages.setdefault(session_id, [])
    
    def get_context(self, session_id: str) -> str:
        self.refresh_from(session_id)
        return self.context.setdefault(session_id, "")
    
Memory = LocalMemory