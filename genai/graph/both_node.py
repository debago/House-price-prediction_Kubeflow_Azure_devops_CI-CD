from genai.rag.rag_service import ask_rag
from genai.services.mlflow_service import get_latest_mlflow_run_summary


def both_node(state):
    rag_answer = ask_rag(state["question"])

    mlflow_answer = get_latest_mlflow_run_summary(
        experiment_name="house-price-prediction"
    )

    return {
        "rag_answer": rag_answer,
        "mlflow_answer": mlflow_answer
    }