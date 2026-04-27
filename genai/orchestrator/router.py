from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from genai.config import OPENAI_API_KEY


ROUTER_PROMPT = """
You are an intelligent router for a California House Price AI platform.

Your job is to decide which agent should answer the user's question.

Available routes:

RAG:
Use this for questions about dataset, features, API, architecture, training pipeline,
documentation, definitions, and general project knowledge.

MLFLOW:
Use this for questions about latest model metrics, latest RMSE, latest MAE,
latest R2 score, latest MLflow run, experiment ID, run ID, and parameters.

ANALYSIS:
Use this when the user asks for interpretation, comparison, diagnosis,
recommendation, model performance explanation, or business/technical summary.

BOTH:
Use this when the question needs both project documentation and live MLflow data.

Return only one word:
RAG
MLFLOW
ANALYSIS
BOTH

Question:
{question}
"""


def route_question(question: str) -> str:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)

    chain = prompt | llm

    response = chain.invoke({
        "question": question
    })

    route = response.content.strip().upper()

    allowed_routes = {"RAG", "MLFLOW", "ANALYSIS", "BOTH"}

    if route not in allowed_routes:
        return "BOTH"

    return route