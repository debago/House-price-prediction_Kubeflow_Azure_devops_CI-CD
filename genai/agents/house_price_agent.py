from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

from genai.rag.rag_service import ask_rag
from genai.tools.mlflow_tool import get_latest_mlflow_run_summary


@tool("house_price_rag_tool")
def house_price_rag_tool(question: str) -> str:
    """
    Use this tool to answer questions about the California house price project
    using project documents, architecture docs, API docs, and model documentation.
    """
    return ask_rag(question)


@tool("mlflow_latest_run_tool")
def mlflow_latest_run_tool(question: str) -> str:
    """
    Use this tool to fetch the latest MLflow run details, including metrics,
    parameters, run ID, and model performance for the house price model.
    """
    return get_latest_mlflow_run_summary(
        experiment_name="house-price-prediction"
    )


def build_house_price_agent():
    agent = Agent(
        role="Enterprise MLOps GenAI Assistant",
        goal=(
            "Answer questions about the California house price prediction project "
            "using RAG documents and MLflow live run metadata."
        ),
        backstory=(
            "You are an AI assistant specialized in MLOps, MLflow, FastAPI, "
            "regression models, and enterprise AI platforms. You can decide whether "
            "to use project documentation, MLflow metadata, or both."
        ),
        tools=[
            house_price_rag_tool,
            mlflow_latest_run_tool
        ],
        verbose=True,
        allow_delegation=False
    )

    return agent


def run_house_price_agent(question: str) -> str:
    agent = build_house_price_agent()

    task = Task(
        description=(
            f"Answer this user question: {question}\n\n"
            "Decide which tool is required. "
            "Use RAG for architecture, API, dataset, and documentation questions. "
            "Use MLflow for latest run, metrics, parameters, and performance questions. "
            "Use both tools if needed."
        ),
        expected_output=(
            "A clear, concise answer. Include whether the answer came from "
            "project documentation, MLflow live metadata, or both."
        ),
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()

    return str(result)


if __name__ == "__main__":
    # question = "What is the latest RMSE of the house price model?"
    question = "What dataset is used?"
    answer = run_house_price_agent(question)
    print(answer)