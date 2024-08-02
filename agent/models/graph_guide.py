import uuid
from typing import List, TypedDict, Union, Literal

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph

from models.tools.llm_connect import LLMConnect
from config import settings


class Message(TypedDict):
    role: Literal["user", "system", "assistant"]
    content: str

def llm():
    llmConnector = LLMConnect(
        api_type=settings.openai_api_type, 
        api_key=settings.openai_api_key, 
        api_base=settings.openai_api_base, 
        deployment_name=settings.openai_deployment_name, 
        org_id=settings.openai_org_id
    )
    return llmConnector.openai(async_client=True)

async def invoke(prompt: str, history: List[Message]) -> str:
    history.append(Message(role="user", content=prompt))
    result = llm().chat.completions.create(messages=history)
    return result

class ConversationState(TypedDict):
    conversation_id: str
    prompt: str
    result: Union[str, None]
    history: List[str]
    
def newConversation(prompt: str) -> ConversationState:
    return {
        "conversation_id": str(uuid.uuid4()),
        "prompt": prompt,
        "result": None,
        "history": [] # TODO add the system prompt in here
    }

async def guide_start_node(state: ConversationState) -> ConversationState:
    state['result'] = await invoke(state['prompt'], state['history'])
    return state

async def guide_start_edge(state: ConversationState) -> Literal["response", "question"]:
    return "response"

async def response(state: ConversationState) -> ConversationState:
    return state


conversation = StateGraph(input=ConversationState, output=ConversationState)
conversation.add_node("guide_start_node", guide_start_node)
conversation.add_node("response", response)
conversation.add_edge(START, "guide_start_node")
conversation.add_edge("response", END)

def start_conversation(prompt: str, say=None) -> ConversationState:
    return newConversation(prompt)