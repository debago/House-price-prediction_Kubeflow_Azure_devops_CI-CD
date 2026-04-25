from langchain_chroma import Chroma
from genai.embeddings.embedding_service import get_embedding_model


VECTOR_DB_PATH = "vector_db/chroma"


def create_vector_store(chunks):
    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    return vector_store


def load_vector_store():
    embeddings = get_embedding_model()

    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )