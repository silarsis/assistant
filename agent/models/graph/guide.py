import uuid
from typing import List, TypedDict, Union, Literal, Optional
import datetime

# from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph

import openai

from models.tools.llm_connect import llm_from_settings
from models.tools import memory
from models.tools.memory import Message
from models.tools import doc_store

from config import settings


class ConfigDict(TypedDict):
    callback: Optional[str]
    hear_thoughts: bool


def llm():
    return llm_from_settings().openai(async_client=True)


async def invoke_llm(messages: List[Message]) -> str:
    try:
        # result = llm_from_settings().openai().chat.completions.create(messages=messages, model=settings.openai_deployment_name)
        result = await llm().chat.completions.create(messages=messages, model=settings.openai_deployment_name)
    except openai.BadRequestError as e:
        print(f"Failed to invoke LLM: {e}")
        return e.message
    except openai.APIConnectionError as e:
        print(f"Failed to connect to OpenAI: {e}")
        return e.message
    return result.choices[0].message.content

class ConversationState(TypedDict):
    conversation_id: str
    character: str
    prompt: str
    result: Union[str, None]
    history_context: Message
    rag_context: str
    history: List[Message]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['conversation_id'] = self['conversation_id'] or str(uuid.uuid4())
        self['result'] = self['result'] or None
        self['history_context'] = self['history_context'] or Message(role="assistant", content="-")
        self['history'] = self['history'] or []
        self['rag_context'] = ''


async def initialise(state: ConversationState, config: ConfigDict) -> ConversationState:
    mem = memory.Memory()
    state['history_context'] = mem.get_context(session_id=state['conversation_id'])
    state['history'] = mem.get_history_for_chatbot(session_id=state['conversation_id'])
    state['result'] = None
    return state

async def ask_llm(state: ConversationState, config: ConfigDict) -> ConversationState:
    user_prompt = Message(role="user", content=state['prompt'])
    state['history'].append(user_prompt)
    await memory.Memory().add_message(user_prompt, session_id=state['conversation_id'])
    messages = [
        Message(role="system", content="Time: " + str(datetime.datetime.now()) + "\n" + state['character']),
        Message(role="assistant", content="Context from docs: " + state['rag_context']),
        state['history_context']
    ] + state['history']
    state['result'] = await invoke_llm(messages)
    return state

async def question(state: ConversationState, config: Optional[dict] = None) -> ConversationState:
    # Send something to the human, then wait for their response
    return state

async def rag_retrieval(state: ConversationState, config: Optional[dict] = None) -> ConversationState:
    # Call the docstore with a retrieval query, dump the results into the state as context for the next query
    state['rag_context'] = '\n'.join(doc_store.DocStore().search_for_phrases(state['prompt']))
    return state

async def reword(state: ConversationState, config: ConfigDict) -> ConversationState:
    # Rephrase the response as per the character - not working yet, shouldn't be used
    messages = [
        Message(role="system", content=state['character']),
        Message(role='system', content="Time: " + str(datetime.datetime.now())),
        Message(role="system", content="Question: ${{input}}"),
        Message(role="system", content="Answer: ${{answer_mesg}}"),
        Message(role="system", content="""
Please respond to the user with this answer. If your chat history or context suggests a better answer, please use that instead.
Check the chat history and context for answers to the question also.
"""),
        Message(role="system", content="Context: {{$context}}"),
    ]
    messages.extend(state['history'])
    state['result'] = await invoke_llm(messages)
    return state

async def check_if_answered(state: ConversationState) -> Literal[END, "reword", "question"]:
    return END


conversation = StateGraph(input=ConversationState, output=ConversationState, config_schema=ConfigDict)

conversation.add_node("initialise", initialise)
conversation.add_node("rag_retrieval", rag_retrieval)
conversation.add_node("ask_llm", ask_llm)
conversation.add_node("question", question)
conversation.add_node("reword", reword)

conversation.add_edge(START, "initialise")
conversation.add_edge("initialise", "rag_retrieval")
conversation.add_edge("rag_retrieval", "ask_llm")
conversation.add_conditional_edges("ask_llm", check_if_answered)
conversation.add_edge("question", "ask_llm") # This should be something else, I think
conversation.add_edge("reword", END)

entry = conversation.compile()


async def invoke(prompt: str, callback=None, hear_thoughts: bool = False, session_id: str="default", character: str="") -> ConversationState:
    result = await entry.ainvoke(
        ConversationState(prompt=prompt, conversation_id=session_id, character=character),
        config=ConfigDict(callback=callback, hear_thoughts=hear_thoughts)
    )
    await memory.Memory().add_message(Message(role="assistant", content=result['result']), session_id=session_id)
    return result['result']