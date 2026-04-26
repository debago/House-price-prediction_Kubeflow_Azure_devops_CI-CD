from crewai import Task


def rag_task(question, agent):
    return Task(
        description=f"Use RAG to answer: {question}",
        expected_output="Relevant information from docs",
        agent=agent
    )


def mlflow_task(question, agent):
    return Task(
        description=f"Fetch latest MLflow metrics for: {question}",
        expected_output="Latest model performance metrics",
        agent=agent
    )


def analysis_task(question, agent):
    return Task(
        description=f"Combine all results and answer clearly: {question}",
        expected_output="Final combined answer",
        agent=agent
    )