from typing import Optional, List, Tuple
import requests
import os
import json

import semantic_kernel as sk

from config import settings

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
    
    def __init__(self, kernel=None):
        print("Local Memory")
        self.context = {}
        self.messages = {}
        self.kernel = kernel
        self.prompt = kernel.create_semantic_function(
            prompt_template=self.template, max_tokens=2000, temperature=0.2, top_p=0.5)
        
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
        with open(f".data/{session_id}.txt", "w") as f:
            f.write(json.dumps(data))
                
    def load(self, session_id: str) -> None:
        # Load messages from file
        try:
            with open(f".data/{session_id}.txt", "r") as f:
                data = json.loads(f.read())
                self.context[session_id] = data['context']
                self.messages[session_id] = data['messages']
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
        ctx = self.kernel.create_new_context()
        ctx['context'] = self.context.setdefault(session_id, "") # get_context refreshes from file
        ctx['history'] = "\n".join([message["content"] for message in contextualise])
        response = await self.prompt(context=ctx)
        self.context[session_id] = response.result
        return response.result
    
    async def add_message(self, role: str, content: str, session_id: str) -> None:
        self.messages.setdefault(session_id, []).append({"role": role, "content": content})
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
        input_list = self.get_history(session_id)
        retval = []
        for i in range(0, len(input_list), 2):
            try:
                retval.append((input_list[i], input_list[i+1]))
            except IndexError:
                retval.append((input_list[-1], ""))
        return retval
    
    def get_context(self, session_id: str) -> str:
        self.refresh_from(session_id)
        return self.context.setdefault(session_id, "")
    
class MotorheadMemory:
    url: str = f"http://{settings.motorhead_host}:{settings.motorhead_port}"
    timeout = 3000
    memory_key = "history"
    context: Optional[str] = None
    
    def __init__(self, kernel=None):
        print("Motorhead memory")
        # We don't use the kernel in here, it's all embedded in the motorhead service
        self.context = {}
        self.messages = {}
    
    def refresh_from(self, session_id: str):
        res = requests.get(
            f"{self.url}/sessions/{session_id}/memory",
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )
        res_data = res.json()
        self.context[session_id] = res_data.get("context", "NONE")
        messages = res_data.get("messages", [])
        messages.reverse()
        self.messages[session_id] = messages # Not strictly thread safe, but not too harmful

    async def add_message(self, role: str, content: str, session_id: str):
        requests.post(
            f"{self.url}/sessions/{session_id}/memory",
            timeout=self.timeout,
            json={
                "messages": [
                    {"role": role, "content": f"{content}"},
                ]
            },
            headers={"Content-Type": "application/json"},
        )
        self.messages.setdefault(session_id, []).append({"role": role, "content": content})
        
    def get_history(self, session_id) -> List[str]:
        if session_id not in self.messages:
            self.refresh_from(session_id)
        return [message["content"] for message in self.messages[session_id]]
    
    def get_formatted_history(self, session_id) -> str:
        history = self.get_history(session_id)
        return "\n".join(history)
    
    def get_context(self, session_id) -> str:
        self.refresh_from(session_id)
        return self.context[session_id]
    
if os.environ.get('MEMORY') == 'motorhead':
    Memory = MotorheadMemory
else:
    Memory = LocalMemory