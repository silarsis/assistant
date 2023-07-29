from models.tools.prompt_template import PromptTemplate

class Planner:
    character = "You are an AI assistant building and executing a plan for multi-step task."
    initial_prompt = """
{{character}}

Given the query below, please produce a plan for how to answer it.
The plan should be 1 to 10 steps long.
Each step in the plan should be a query that can be answered by another AI assistant,
and the answer to each step will be added to a history for the next step.
The final step should summarise an answer from the history provided.
Each step can also use the following tools:

{{tools}}

Query:
{{await 'query'}}

Plan:
{{gen 'plan'}}
"""
    queue_prompt = """
{{character}}

Given the query below and the history of results from previous steps, please execute the next step in the plan.

Query:
{{await 'query'}}

History:
{{await 'history'}}

Step:
{{await 'step'}}

Result:
{{gen 'result'}}
"""

    def __init__(self, llm):
        self.llm = llm
        self._query = ""
        self._queue = []
        self._history = []
        self._initial_prompt_templates = PromptTemplate(self.character, self.initial_prompt)
        self._step_prompt_templates = PromptTemplate(self.character, self.queue_prompt)
        
    def _run_queue(self) -> str: # Need session id here
        if self._queue:
            step = self._queue.pop(0)
            # Need to think about how to call the tools from here
            res = self._step_prompt_templates.get(self._session_id, self.llm)(query=self._query, history=self._history, step=step)
            response = res['result']
            self._history.append(f"Step: {step}\nResponse: {response}")
            print(self._history[-1])
            return self._run_queue()
        else:
            return self._history[-1]
        
    # a queue per session
    def run(self, tools: list, query: str, session_id: str = 'client') -> str:
        self._query =  query
        self._tools = tools
        self._session_id = session_id
        tools_str = '\n'.join([f"{tool.name} ({tool.description})\n" for tool in tools])
        # Run the first query as a "please create a plan as a list of queries to run"
        res = self._initial_prompt_templates.get(session_id, self.llm)(query=query, tools=tools_str)
        plan = res['plan']
        # Then, take each item in the list and add it to self.queue
        self._queue = plan.split('\n')
        # Then, call the queue runner on a loop until the queue is empty and return the last response
        return self._run_queue()
    
"""
A planner should be able to keep a queue of things to do,
with each thing potentially being multiple things in parallel.
It will need references to inputs and outputs from the other tasks,
and to any tools that are available.

The idea is that we analyse a query, determine if it needs planning, 
and if so, we add it to the queue. Then, we run the queue, and return
the results. But the queue should be able to add to itself also.
"""
