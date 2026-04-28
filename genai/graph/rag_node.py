from genai.rag.rag_service import ask_rag


def rag_node(state):
    answer = ask_rag(state["question"])

    return {
        "rag_answer": answer
    }