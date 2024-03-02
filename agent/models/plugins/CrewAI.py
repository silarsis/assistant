from typing import Any, List

from config import settings

from crewai import Agent, Task, Crew, Process
from semantic_kernel.plugin_definition import kernel_function, kernel_function_context_parameter
from semantic_kernel.orchestration.kernel_context import KernelContext

from pydantic import BaseModel

class CrewAIPlugin(BaseModel):
    kernel: Any = None
    agents: List[Agent] = []
    tasks: List[Task] = []
    
    def _create_agents(self, role: str, goal: str, backstory: str) -> None:
        self.agents = []
        for crew in settings.crew:
            self.agents.append(
                Agent(
                    role=crew.role, 
                    goal=crew.goal, 
                    backstory=crew.backstory, 
                    allow_delegation=True, 
                    llm=self.kernel, 
                    verbose=True
                )
            )
        
    def _create_task(self, description: str, expected_outcome: str) -> None:
        self.tasks = [
            Task(
                description=description,
                expected_outcome=expected_outcome
            )
        ]
    
    def _setup_crew(self) -> None:
        self._create_agents()
        self._crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            manager_llm=self.kernel,
            process=Process.hierarchical,
            verbose=2
        )
    
    @kernel_function(
        description="Complex problems requiring a crew of agents",
        name="ask_crew"
    )
    @kernel_function_context_parameter(
        name='task',
        description='The task to undertake'
    )
    @kernel_function_context_parameter(
        name='outcome',
        description='The expected outcome of the task'
    )
    def ask_crew(self, context: KernelContext) -> str:
        self._create_task(context['task'], context['outcome'])
        self._setup_crew()
        self._crew.kickoff()
        

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