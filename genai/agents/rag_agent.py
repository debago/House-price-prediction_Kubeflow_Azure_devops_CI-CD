from crewai import Agent
from genai.tools.tools import rag_tool


def create_rag_agent():
    return Agent(
        role="RAG Specialist",
        goal="Answer questions using project documentation",
        backstory="Expert in retrieving knowledge from vector database",
        tools=[rag_tool],
        verbose=True
    )