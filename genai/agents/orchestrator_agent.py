from crewai import Agent


def create_orchestrator_agent():
    return Agent(
        role="AI Platform Orchestrator",
        goal="Coordinate specialist agents to answer house price model questions.",
        backstory=(
            "You are a manager agent. You understand whether a question requires "
            "project documentation, MLflow metrics, or both. You coordinate other "
            "specialist agents and ensure the final response is clear."
        ),
        verbose=True,
        allow_delegation=True
    )