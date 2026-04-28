from typing import TypedDict, Optional


class HousePriceGraphState(TypedDict):
    question: str
    route: Optional[str]
    rag_answer: Optional[str]
    mlflow_answer: Optional[str]
    final_answer: Optional[str]