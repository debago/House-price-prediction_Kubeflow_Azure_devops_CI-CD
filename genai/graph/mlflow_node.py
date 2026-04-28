from genai.services.mlflow_service import get_latest_mlflow_run_summary


def mlflow_node(state):
    answer = get_latest_mlflow_run_summary(
        experiment_name="house-price-prediction"
    )

    return {
        "mlflow_answer": answer
    }