from crewai.tools import tool
from genai.services.mlflow_service import get_latest_mlflow_run_summary

@tool("mlflow_latest_run_tool")
def mlflow_latest_run_tool(question: str) -> str:
    """
    Use this tool to fetch the latest MLflow run details, including metrics,
    parameters, run ID, and model performance for the house price model.
    """
    return get_latest_mlflow_run_summary(
        experiment_name="house-price-prediction"
    )