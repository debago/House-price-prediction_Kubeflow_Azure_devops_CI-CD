from crewai.tools import tool
import mlflow
from genai.rag.rag_service import ask_rag
from genai.config import MLFLOW_TRACKING_URI


# Set MLflow URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@tool("rag_tool")
def rag_tool(question: str) -> str:
    """
    Use this tool to answer questions from project documentation
    like dataset, architecture, API, and model overview.
    """
    return ask_rag(question)


@tool("mlflow_tool")
def mlflow_tool(question: str) -> str:
    """
    Use this tool to fetch latest MLflow run metrics
    like RMSE, MAE, R2 score, and parameters.
    """

    experiment = mlflow.get_experiment_by_name("Default")

    if experiment is None:
        return "No MLflow experiment found."

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs.empty:
        return "No MLflow runs found."

    latest_run = runs.iloc[0]

    metrics = {
        col.replace("metrics.", ""): latest_run[col]
        for col in runs.columns
        if col.startswith("metrics.")
    }

    return f"Latest MLflow Metrics: {metrics}"