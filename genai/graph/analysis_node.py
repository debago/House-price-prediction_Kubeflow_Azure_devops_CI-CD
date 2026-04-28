from langchain_core.prompts import ChatPromptTemplate
from genai.config import llm


ANALYSIS_PROMPT = """
You are a senior MLOps and GenAI analyst.

Answer the user question using the available information.

Question:
{question}

RAG Answer:
{rag_answer}

MLflow Answer:
{mlflow_answer}

Rules:
- If only RAG answer is available, answer from RAG.
- If only MLflow answer is available, answer from MLflow.
- If both are available, combine both clearly.
- Do not invent missing values.
"""


def analysis_node(state):
    rag_answer = state.get("rag_answer") or "Not available"
    mlflow_answer = state.get("mlflow_answer") or "Not available"

    prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
    chain = prompt | llm

    response = chain.invoke({
        "question": state["question"],
        "rag_answer": rag_answer,
        "mlflow_answer": mlflow_answer
    })

    return {
        "final_answer": response.content
    }