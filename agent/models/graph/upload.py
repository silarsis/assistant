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
    session_id: str


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
    character: str
    prompt: str
    result: Union[str, None]
    history_context: Message
    rag_context: str
    history: List[Message]
    file: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['result'] = self['result'] or None
        self['history_context'] = self['history_context'] or Message(role="assistant", content="-")
        self['history'] = self['history'] or []
        self['file'] = self['prompt'] or ''


STORED_STATES = {}

def save_state(session_id: str, state: ConversationState) -> None:
    STORED_STATES[session_id] = state

def load_state(session_id: str) -> ConversationState:
    return STORED_STATES.get(session_id, None)


async def initialise(state: ConversationState, config: ConfigDict) -> ConversationState:
    stored_state = load_state(config['metadata']['session_id'])
    if stored_state:
        # Replace existing state with stored one, and carry on with the conversation
        return stored_state
    mem = memory.Memory(config['metadata']['session_id'])
    state['history_context'] = mem.get_context()
    state['history'] = mem.get_history_for_chatbot()
    state['result'] = None
    return state

async def decide_action(state: ConversationState, config: ConfigDict) -> ConversationState:
    # Take the file upload (already converted to text), and check history to see what to do with it
    messages = [
        Message(role="system", content="Time: " + str(datetime.datetime.now()) + "\n" + state['character']),
        state['history_context'],
    ] + state['history'] + [
        Message(role="user", content="Based on the history of the conversation so far, is it clear what to do with the file contents? If it is, then generate a query that matches the intent. If it is not clear, then simply say 'ask user'")
    ]
    result = await invoke_llm(messages)
    if result == 'ask user':
        state['result'] = "Thanks for the file, what would you like to know about it?"
    else:
        state['result'] = result
    return state


async def ask_llm(state: ConversationState, config: ConfigDict) -> ConversationState:
    user_prompt = Message(role="user", content=state['result'])
    await memory.Memory(config['metadata']['session_id']).add_message(user_prompt)
    messages = [
        Message(role="system", content="Time: " + str(datetime.datetime.now()) + "\n" + state['character']),
        state['history_context']
    ] + state['history'] + [ user_prompt ]
    state['result'] = await invoke_llm(messages)
    return state

async def question(state: ConversationState, config: Optional[dict] = None) -> ConversationState:
    # Send something to the human, then wait for their response
    # I think this means store the current state somewhere, and return as though we're done, then
    # on the next query refresh the state and carry on.
    save_state(config['metadata']['session_id'], state)
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

async def action_choice(state: ConversationState) -> Literal["ask_llm", "question", "reword"]:
    if state['result'].lower() == "Thanks for the file, what would you like to know about it?":
        return "question"
    return "ask_llm"


conversation = StateGraph(input=ConversationState, output=ConversationState, config_schema=ConfigDict)

conversation.add_node("initialise", initialise)
conversation.add_node("decide_action", decide_action)
conversation.add_node("ask_llm", ask_llm)
conversation.add_node("question", question)
conversation.add_node("reword", reword)

conversation.add_edge(START, "initialise")
conversation.add_edge("initialise", "decide_action")
conversation.add_conditional_edges("decide_action", action_choice)
conversation.add_edge("ask_llm", END)
conversation.add_edge("question", END)
conversation.add_edge("reword", END)

entry = conversation.compile()


async def invoke(prompt: str, callback=None, hear_thoughts: bool = False, session_id: str="default", character: str="") -> ConversationState:
    result = await entry.ainvoke(
        ConversationState(prompt=prompt, character=character),
        config=ConfigDict(callback=callback, hear_thoughts=hear_thoughts, session_id=session_id)
    )
    await memory.Memory(session_id).add_message(Message(role="assistant", content=result['result']))
    return result['result']