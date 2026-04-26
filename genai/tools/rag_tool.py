from crewai.tools import tool
from genai.rag.rag_service import ask_rag


@tool("rag_tool")
def rag_tool(question: str) -> str:
    """
    Use this tool to answer questions from project documentation
    like dataset, architecture, API, and model overview.
    """
    return ask_rag(question)
