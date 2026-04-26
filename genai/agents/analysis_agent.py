from crewai import Agent


def create_analysis_agent():
    return Agent(
        role="Senior ML Analyst",
        goal="Combine documentation and MLflow outputs into clear insights.",
        backstory="Expert in model evaluation, regression metrics, and MLOps explanation.",
        verbose=True
    )