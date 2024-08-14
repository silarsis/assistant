from typing import Optional, List, Tuple, TypedDict, Literal
import os
import json

from models.tools.llm_connect import llm_from_settings
from config import settings

class Message(TypedDict):
    role: Literal["user", "system", "assistant"]
    content: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.role:
            raise KeyError("Role must be one of 'user', 'system', or 'assistant'")
        if not self.content:
            raise ValueError("Content must be a string")


class Memory:
    context: Optional[Message] = None

    def __init__(self):
        self.context = {}
        self.messages = {}
        self.llm = llm_from_settings().openai(async_client=True)

    def refresh_from(self, session_id: str) -> None:
        self.load(session_id)
        self.context.setdefault(session_id, Message(role="assistant", content=""))
        self.messages.setdefault(session_id, [])

    def save(self, session_id: str) -> None:
        # Save messages to a file, fix the error if the dir doesn't exist
        os.makedirs(".data", exist_ok=True)
        data = {
            'context': self.context.setdefault(session_id, Message(role="assistant", content="")),
            'messages': self.messages.get(session_id, [])
        }
        with open(f".data/{session_id}.json", "w") as f:
            f.write(json.dumps(data))

    def load(self, session_id: str) -> None:
        # Load messages from file
        try:
            with open(f".data/{session_id}.json", "r") as f:
                data = json.loads(f.read())
                try:
                    self.context[session_id] = Message(**data['context'] or "-")
                except TypeError:
                    self.context[session_id] = Message(role="assistant", content=data['context'])
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
        # call the llm, get the results, update the context
        context=self.context.setdefault(session_id, Message(role="assistant", content=""))
        prompt = Message(role="system", content="Summarise the following conversation history, taking the existing context into account")
        response = await self.llm.chat.completions.create(messages=[prompt, context] + contextualise, model=settings.openai_deployment_name)
        self.context[session_id] = Message(role="assistant", content=response.choices[0].message.content)
        return self.context[session_id]

    async def add_message(self, mesg: Message, session_id: str) -> None:
        self.messages.setdefault(session_id, []).append(Message(role=mesg['role'], content=mesg['content']))
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

    def get_history_for_chatbot(self, session_id: str) -> List[Message]:
        if session_id not in self.messages:
            self.refresh_from(session_id)
        return self.messages.setdefault(session_id, [])

    def get_context(self, session_id: str) -> Message:
        self.refresh_from(session_id)
        return self.context.setdefault(session_id, Message(role="assistant", content=""))