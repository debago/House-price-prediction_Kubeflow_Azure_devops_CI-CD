from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from genai.rag.prompt_templates import RAG_PROMPT
from genai.vector_store.vector_store_service import load_vector_store
from genai.tools.mlflow_tool import get_latest_mlflow_run_summary


def ask_rag(question: str) -> str:
    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(question)

    doc_context = "\n\n".join(doc.page_content for doc in docs)

    mlflow_context = get_latest_mlflow_run_summary(
        experiment_name="house-price-prediction"
    )

    final_context = f"""
Document Context:
{doc_context}

MLflow Live Context:
{mlflow_context}
"""

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": final_context,
        "question": question
    })

    return response

# def ask_rag(question: str) -> str:
#     vector_store = load_vector_store()

#     retriever = vector_store.as_retriever(
#         search_kwargs={"k": 3}
#     )

#     docs = retriever.invoke(question)

#     context = "\n\n".join(doc.page_content for doc in docs)

#     prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

#     llm = ChatOpenAI(
#         model="gpt-4o-mini",
#         temperature=0
#     )

#     chain = prompt | llm | StrOutputParser()

#     response = chain.invoke({
#         "context": context,
#         "question": question
#     })

#     return response


if __name__ == "__main__":
    # answer = ask_rag("What is the registered MLflow model name?")
    answer = ask_rag("What dataset is used?")
    print(answer)