from crewai import Agent
from genai.tools.rag_tool import rag_tool
from genai.tools.mlflow_tool import mlflow_latest_run_tool


def rag_agent():
    return Agent(
        role="RAG Specialist",
        goal="Answer questions using project documentation",
        backstory="Expert in retrieving knowledge from vector database",
        tools=[rag_tool],
        verbose=True
    )


def mlflow_agent():
    return Agent(
        role="MLflow Expert",
        goal="Provide latest model metrics",
        backstory="Expert in MLflow tracking and model performance",
        tools=[mlflow_latest_run_tool],
        verbose=True
    )


def analysis_agent():
    return Agent(
        role="Senior ML Analyst",
        goal="Combine outputs and give final insights",
        backstory="Expert in analyzing ML models and explaining clearly",
        verbose=True
    )