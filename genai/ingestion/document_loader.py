from pathlib import Path

from langchain_community.document_loaders import TextLoader


DOCS_PATH = "docs"


def load_documents():
    documents = []

    for file_path in Path(DOCS_PATH).glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents.extend(loader.load())

    return documents