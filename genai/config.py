import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # loads .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if not MLFLOW_TRACKING_URI:
    raise ValueError("MLFLOW_TRACKING_URI not set")

# 👇 Central LLM object
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)