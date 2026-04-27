from crewai import Agent
# from genai.tools.tools import mlflow_tool
from genai.tools.mlflow_tool import mlflow_latest_run_tool


def create_mlflow_agent():
    return Agent(
        role="MLflow Metrics Expert",
        goal="Provide latest model metrics and run details",
        backstory="Expert in MLflow experiment tracking and model performance",
        tools=[mlflow_latest_run_tool],
        verbose=True
    )