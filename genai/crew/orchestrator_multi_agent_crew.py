from crewai import Crew, Process

from genai.agents.orchestrator_agent import create_orchestrator_agent
from genai.agents.rag_agent import create_rag_agent
from genai.agents.mlflow_agent import create_mlflow_agent
from genai.agents.analysis_agent import create_analysis_agent

from genai.tasks.orchestrator_tasks import (
    create_orchestrator_task,
    create_rag_task,
    create_mlflow_task,
    create_analysis_task,
)


def run_orchestrator_multi_agent_crew(question: str):
    orchestrator = create_orchestrator_agent()
    rag_agent = create_rag_agent()
    mlflow_agent = create_mlflow_agent()
    analysis_agent = create_analysis_agent()

    orchestrator_task = create_orchestrator_task(question, orchestrator)
    rag_task = create_rag_task(question, rag_agent)
    mlflow_task = create_mlflow_task(question, mlflow_agent)
    analysis_task = create_analysis_task(question, analysis_agent)

    crew = Crew(
        agents=[
            orchestrator,
            rag_agent,
            mlflow_agent,
            analysis_agent
        ],
        tasks=[
            orchestrator_task,
            rag_task,
            mlflow_task,
            analysis_task
        ],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    return result


if __name__ == "__main__":
    question = "Explain the latest RMSE of the California house price model and what it means."
    result = run_orchestrator_multi_agent_crew(question)
    print(result)