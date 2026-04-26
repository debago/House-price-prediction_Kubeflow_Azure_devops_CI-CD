from crewai import Agent
from genai.tools.tools import mlflow_tool


def create_mlflow_agent():
    return Agent(
        role="MLflow Metrics Expert",
        goal="Provide latest model metrics and run details",
        backstory="Expert in MLflow experiment tracking and model performance",
        tools=[mlflow_tool],
        verbose=True
    )