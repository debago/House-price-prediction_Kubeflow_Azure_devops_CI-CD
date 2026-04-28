from langchain_core.prompts import ChatPromptTemplate
from genai.config import llm


ROUTER_PROMPT = """
You are a router for a California House Price AI system.

Decide which path should answer the question.

Routes:

RAG:
Use for dataset, features, API, architecture, training pipeline,
definitions, documentation, and general project knowledge.

MLFLOW:
Use for latest run, latest RMSE, latest MAE, latest R2,
MLflow metrics, parameters, experiment, and run ID.

BOTH:
Use when the question needs both documentation and latest MLflow values,
or asks for model performance explanation.

Return only one word:
RAG
MLFLOW
BOTH

Question:
{question}
"""


def router_node(state):
    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
    chain = prompt | llm

    response = chain.invoke({
        "question": state["question"]
    })

    route = response.content.strip().upper()

    if route not in {"RAG", "MLFLOW", "BOTH"}:
        route = "BOTH"

    return {
        "route": route
    }


def route_decision(state):
    return state["route"]