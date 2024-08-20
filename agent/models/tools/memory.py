from typing import Optional, List, TypedDict, Literal
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
    messages: List[Message] = None

    def __init__(self, session_id: str = "static"):
        self.session_id = session_id
        self.llm = llm_from_settings().openai(async_client=True)
        self.load()

    def save(self) -> None:
        # Save messages to a file, fix the error if the dir doesn't exist
        os.makedirs(".data", exist_ok=True)
        data = {
            'context': self.context,
            'messages': self.messages
        }
        with open(f".data/{self.session_id}.json", "w") as f:
            f.write(json.dumps(data))

    def load(self) -> None:
        # Load messages from file
        self.context = Message(role="assistant", content="-")
        self.messages = []
        try:
            with open(f".data/{self.session_id}.json", "r") as f:
                data = json.loads(f.read())
                if data.get('context'):
                    self.context = Message(**data['context'])
                else:
                    self.context = Message(role="assistant", content="-")
                self.messages = [Message(**m) for m in data['messages']]
        except FileNotFoundError:
            print("No memory, starting from scratch")
        except json.decoder.JSONDecodeError as e:
            print(f"Failed to decode memory: {e}")

    async def _summarise(self) -> str:
        contextualise = self.messages[:-10]
        if not contextualise:
            return
        self.messages = self.messages[-10:]
        # call the llm, get the results, update the context
        prompt = Message(role="system", content="Summarise the following conversation history, taking the existing context into account")
        response = await self.llm.chat.completions.create(messages=[prompt, self.context] + contextualise, model=settings.openai_deployment_name)
        self.context = Message(role="assistant", content=response.choices[0].message.content)
        return self.context

    async def add_message(self, mesg: Message) -> None:
        self.messages.append(mesg)
        if len(self.messages) > 20:
            await self._summarise()
        self.save()

    def get_history(self) -> List[str]:
        return [message["content"] for message in self.messages]

    def get_formatted_history(self) -> str:
        history = self.get_history()
        return "\n".join(history)

    def get_history_for_chatbot(self) -> List[Message]:
        return self.messages

    def get_context(self) -> Message:
        return self.context