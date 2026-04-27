from crewai import Crew, Process
from genai.agents.agents import rag_agent, mlflow_agent, analysis_agent
from genai.tasks.tasks import rag_task, mlflow_task, analysis_task
from genai.config import llm


def run_crew(question: str):

    # Create agents
    r_agent = rag_agent()
    m_agent = mlflow_agent()
    a_agent = analysis_agent()

    # Create tasks
    t1 = rag_task(question, r_agent)
    t2 = mlflow_task(question, m_agent)
    t3 = analysis_task(question, a_agent)

    # Crew execution
    crew = Crew(
        agents=[r_agent, m_agent, a_agent],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        llm=llm,
        verbose=True
    )

    return crew.kickoff()


if __name__ == "__main__":
    question = "Explain model performance and latest RMSE"
    result = run_crew(question)
    print(result)