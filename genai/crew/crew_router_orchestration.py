from crewai import Crew, Process

from genai.crew.crew import run_crew
from genai.orchestrator.router import route_question
from genai.agents.rag_agent import create_rag_agent
from genai.agents.mlflow_agent import create_mlflow_agent
from genai.agents.analysis_agent import create_analysis_agent
from genai.tasks.tasks import rag_task, mlflow_task, analysis_task


def run_orchestrator(question: str):
    route = route_question(question)

    print(f"Selected route: {route}")

    rag_agent = create_rag_agent()
    mlflow_agent = create_mlflow_agent()
    analysis_agent = create_analysis_agent()

    tasks = []
    agents = []

    if route == "RAG":
        agents.append(rag_agent)
        tasks.append(rag_task(question, rag_agent))

    elif route == "MLFLOW":
        agents.append(mlflow_agent)
        tasks.append(mlflow_task(question, mlflow_agent))

    elif route == "ANALYSIS":
        agents.append(analysis_agent)
        tasks.append(analysis_task(question, analysis_agent))

    elif route == "BOTH":
        agents.extend([rag_agent, mlflow_agent, analysis_agent])
        tasks.extend([
            rag_task(question, rag_agent),
            mlflow_task(question, mlflow_agent),
            analysis_task(question, analysis_agent)
        ])

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    return result


if __name__ == "__main__":
    question = " what is RMSE"
    result = run_orchestrator(question)
    print(result)
    print(f"Route selected: {route_question(question)}")