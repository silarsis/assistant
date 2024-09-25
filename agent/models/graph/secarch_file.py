from typing import Literal, Optional
import datetime

# from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph

from models.tools import memory
from models.tools.memory import Message
from models.tools import doc_store

from models.graph.guide import ConfigDict, ConversationState, invoke_llm


class SecArchState(ConversationState):
    rfc: str
    
    def __init__(self, rfc: str, **kwargs):
        super().__init__(**kwargs)
        self.rfc = rfc


STORED_STATES = {}

def save_state(session_id: str, state: SecArchState) -> None:
    STORED_STATES[session_id] = state

def load_state(session_id: str) -> SecArchState:
    return STORED_STATES.get(session_id, None)


async def initialise(state: SecArchState, config: ConfigDict) -> SecArchState:
    stored_state = load_state(config['metadata']['session_id'])
    if stored_state:
        # Replace existing state with stored one, and carry on with the conversation
        return stored_state
    mem = memory.Memory(config['metadata']['session_id'])
    state['history_context'] = mem.get_context()
    state['history'] = mem.get_history_for_chatbot()
    state['result'] = None
    return state

async def decide_action(state: SecArchState, config: ConfigDict) -> SecArchState:
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


async def ask_llm(state: SecArchState, config: ConfigDict) -> SecArchState:
    user_prompt = Message(role="user", content=state['result'])
    await memory.Memory(config['metadata']['session_id']).add_message(user_prompt)
    messages = [
        Message(role="system", content="Time: " + str(datetime.datetime.now()) + "\n" + state['character']),
        state['history_context']
    ] + state['history'] + [ user_prompt ]
    state['result'] = await invoke_llm(messages)
    return state

async def question(state: SecArchState, config: Optional[dict] = None) -> SecArchState:
    # Send something to the human, then wait for their response
    # I think this means store the current state somewhere, and return as though we're done, then
    # on the next query refresh the state and carry on.
    save_state(config['metadata']['session_id'], state)
    return state

async def rag_retrieval(state: SecArchState, config: Optional[dict] = None) -> SecArchState:
    # Call the docstore with a retrieval query, dump the results into the state as context for the next query
    state['rag_context'] = '\n'.join(doc_store.DocStore().search_for_phrases(state['prompt']))
    return state

async def reword(state: SecArchState, config: ConfigDict) -> SecArchState:
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

async def action_choice(state: SecArchState) -> Literal["ask_llm", "question", "reword"]:
    if state['result'].lower() == "Thanks for the file, what would you like to know about it?":
        return "question"
    return "ask_llm"


conversation = StateGraph(input=SecArchState, output=SecArchState, config_schema=ConfigDict)

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


# This guide is designed to take an uploaded file, and perform a secarch analysis on it.
async def invoke(rfc: str, callback=None, hear_thoughts: bool = False, session_id: str="default", character: str="") -> SecArchState:
    result = await entry.ainvoke(
        SecArchState(prompt=rfc, character=character),
        config=ConfigDict(callback=callback, hear_thoughts=hear_thoughts, session_id=session_id)
    )
    await memory.Memory(session_id).add_message(Message(role="assistant", content=result['result']))
    return result['result']