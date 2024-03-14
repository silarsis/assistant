from typing import Any, List, Annotated, Union

from config import settings

from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool
from semantic_kernel.functions.kernel_function_decorator import kernel_function

from models.tools.llm_connect import LLMConnect, AzureChatOpenAI, ChatOpenAI

from pydantic import BaseModel

## Monkey patch telemetry out
from crewai.telemetry import Telemetry

def noop(*args, **kwargs):
    pass

for attr in dir(Telemetry):
    if callable(getattr(Telemetry, attr)) and not attr.startswith("__"):
        setattr(Telemetry, attr, noop)
## End monkey patch

class CrewAIPlugin(BaseModel):
    client: Any = None
    kernel: Any = None
    llm: Union[ChatOpenAI,AzureChatOpenAI] = None
    agents: List[Agent] = []
    tasks: List[Task] = []
    
    def __init__(self, kernel=None):
        super().__init__()
        self.kernel = kernel
        self._crew_tools = list(self._tools())
        self.llm = LLMConnect(
            api_type=settings.openai_api_type, 
            api_key=settings.openai_api_key, 
            api_base=settings.openai_api_base, 
            deployment_name=settings.openai_deployment_name, 
            org_id=settings.openai_org_id
        ).langchain()
        
    def step_callback(self, *args, **kwargs):
        pass
        
    def _tools(self):
        " Return a list of crewai-usable tools from the semantic_kernel plugins "
        for plugin in self.kernel.plugins:
            for function in plugin.functions:
                class CustomTool(BaseTool):
                    name: str = function
                    description: str = plugin.functions[function].description
                    def _run(self, argument: str) -> str:
                        return self.kernel.invoke(plugin.functions[function], argument)
                yield CustomTool
                
        
    def _create_agents(self) -> None:
        self.agents = []
        for crew in settings.crew:
            self.agents.append(
                Agent(
                    role=crew.role, 
                    goal=crew.goal, 
                    backstory=crew.backstory, 
                    allow_delegation=True, 
                    tools=self._crew_tools,
                    llm=self.llm, 
                    verbose=True
                )
            )
        
    def _create_task(self, description: str, expected_output: str) -> None:
        self.tasks = [
            Task(
                description=description,
                expected_output=expected_output
            )
        ]
    
    def _setup_crew(self) -> None:
        self._create_agents()
        # if settings.hear_thoughts:
        #     step_callback = self.step_callback
        # else:
        #     step_callback = None
        self._crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            manager_llm=self.llm,
            process=Process.hierarchical,
            verbose=2
        )
    
    @kernel_function(name="ask_crew", description="Ask a crew to undertake a task")
    def ask_crew(self, task: Annotated[str, "The task to undertake"], outcome: Annotated[str, "The expected outcome of the task"]) -> str:
        " Ask a crew to undertake a task "
        self._create_task(task, outcome)
        self._setup_crew()
        return self._crew.kickoff()
        

# # Define your agents with roles and goals
# researcher = Agent(
#   role='Senior Research Analyst',
#   goal='Uncover cutting-edge developments in AI and data science',
#   backstory="""You work at a leading tech think tank.
#   Your expertise lies in identifying emerging trends.
#   You have a knack for dissecting complex data and presenting actionable insights.""",
#   verbose=True,
#   allow_delegation=False,
#   tools=[search_tool]
#   # You can pass an optional llm attribute specifying what mode you wanna use.
#   # It can be a local model through Ollama / LM Studio or a remote
#   # model like OpenAI, Mistral, Antrophic or others (https://docs.crewai.com/how-to/LLM-Connections/)
#   #
#   # import os
#   # os.environ['OPENAI_MODEL_NAME'] = 'gpt-3.5-turbo'
#   #
#   # OR
#   #
#   # from langchain_openai import ChatOpenAI
#   # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7)
# )
# writer = Agent(
#   role='Tech Content Strategist',
#   goal='Craft compelling content on tech advancements',
#   backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
#   You transform complex concepts into compelling narratives.""",
#   verbose=True,
#   allow_delegation=True
# )

# # Create tasks for your agents
# task1 = Task(
#   description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
#   Identify key trends, breakthrough technologies, and potential industry impacts.""",
#   expected_output="Full analysis report in bullet points",
#   agent=researcher
# )

# task2 = Task(
#   description="""Using the insights provided, develop an engaging blog
#   post that highlights the most significant AI advancements.
#   Your post should be informative yet accessible, catering to a tech-savvy audience.
#   Make it sound cool, avoid complex words so it doesn't sound like AI.""",
#   expected_output="Full blog post of at least 4 paragraphs",
#   agent=writer
# )

# # Instantiate your crew with a sequential process
# crew = Crew(
#   agents=[researcher, writer],
#   tasks=[task1, task2],
#   verbose=2, # You can set it to 1 or 2 to different logging levels
# )

# # Get your crew to work!
# result = crew.kickoff()

# print("######################")
# print(result)