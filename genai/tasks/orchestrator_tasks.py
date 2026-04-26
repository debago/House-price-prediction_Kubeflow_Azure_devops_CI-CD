from crewai import Task


def create_orchestrator_task(question: str, orchestrator_agent) -> Task:
    return Task(
        description=(
            f"User question:\n{question}\n\n"
            "Coordinate the specialist agents to answer this question.\n\n"
            "Decision rules:\n"
            "- Delegate to RAG Specialist for dataset, features, architecture, API, "
            "training pipeline, and documentation questions.\n"
            "- Delegate to MLflow Metrics Expert for latest RMSE, MAE, R2, run ID, "
            "parameters, and model performance questions.\n"
            "- Delegate to Senior ML Analysis Expert to combine results into a final answer.\n"
            "- Use only the necessary agents."
        ),
        expected_output=(
            "A clear final answer mentioning whether the response is based on "
            "RAG documentation, MLflow metadata, or both."
        ),
        agent=orchestrator_agent
    )


def create_rag_task(question: str, rag_agent) -> Task:
    return Task(
        description=(
            f"Use RAG documentation to answer this question:\n{question}"
        ),
        expected_output="Relevant answer from project documentation.",
        agent=rag_agent
    )


def create_mlflow_task(question: str, mlflow_agent) -> Task:
    return Task(
        description=(
            f"Use MLflow to fetch latest run metrics for this question:\n{question}"
        ),
        expected_output="Latest MLflow metrics, parameters, and run details.",
        agent=mlflow_agent
    )


def create_analysis_task(question: str, analysis_agent) -> Task:
    return Task(
        description=(
            f"Combine previous agent outputs and answer clearly:\n{question}"
        ),
        expected_output="Final consolidated answer.",
        agent=analysis_agent
    )