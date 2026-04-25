
from genai.ingestion.document_splitter import split_documents
from genai.vector_store.vector_store_service import create_vector_store
from genai.ingestion.document_loader import load_documents


def run_ingestion():
    documents = load_documents()

    if not documents:
        raise ValueError("No documents found inside docs folder.")

    chunks = split_documents(documents)

    create_vector_store(chunks)

    print(f"Successfully ingested {len(chunks)} chunks into vector DB.")


if __name__ == "__main__":
    run_ingestion()