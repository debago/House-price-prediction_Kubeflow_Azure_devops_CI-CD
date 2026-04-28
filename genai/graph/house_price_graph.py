from langgraph.graph import StateGraph, END

from genai.graph.state import HousePriceGraphState
from genai.graph.router_node import router_node, route_decision
from genai.graph.rag_node import rag_node
from genai.graph.mlflow_node import mlflow_node
from genai.graph.both_node import both_node
from genai.graph.analysis_node import analysis_node


def build_house_price_graph():
    graph = StateGraph(HousePriceGraphState)

    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("mlflow", mlflow_node)
    graph.add_node("both", both_node)
    graph.add_node("analysis", analysis_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "RAG": "rag",
            "MLFLOW": "mlflow",
            "BOTH": "both"
        }
    )

    graph.add_edge("rag", "analysis")
    graph.add_edge("mlflow", "analysis")
    graph.add_edge("both", "analysis")
    graph.add_edge("analysis", END)

    return graph.compile()


def run_house_price_graph(question: str):
    app = build_house_price_graph()

    result = app.invoke({
        "question": question,
        "route": None,
        "rag_answer": None,
        "mlflow_answer": None,
        "final_answer": None
    })

    return result["final_answer"]


if __name__ == "__main__":
    questions = [
        "What dataset is used?",
        "What is latest RMSE?",
        "Explain model performance using latest run"
    ]

    for question in questions:
        print("\nQUESTION:", question)
        print(run_house_price_graph(question))