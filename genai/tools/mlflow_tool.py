import mlflow


def get_latest_mlflow_run_summary(experiment_name: str = "Default") -> str:
    """
    Fetch latest MLflow run metrics and params.
    Returns summary text that can be passed to RAG context.
    """

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        return f"No MLflow experiment found with name: {experiment_name}"

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs.empty:
        return f"No MLflow runs found for experiment: {experiment_name}"

    latest_run = runs.iloc[0]

    metrics = {
        col.replace("metrics.", ""): latest_run[col]
        for col in runs.columns
        if col.startswith("metrics.")
    }

    params = {
        col.replace("params.", ""): latest_run[col]
        for col in runs.columns
        if col.startswith("params.")
    }

    return f"""
Latest MLflow Run Summary:

Run ID: {latest_run["run_id"]}

Metrics:
{metrics}

Parameters:
{params}
"""