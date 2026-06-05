def build_index(files):
    """
    files: list of Streamlit UploadedFile objects
    Person A implements this — rag/ingestion.py + rag/embeddings.py
    Returns: None
    """
    raise NotImplementedError

def ask(query: str) -> dict:
    """
    Person A implements this — RetrievalQA chain
    Returns: {"answer": str, "sources": [{"doc": str, "page": int}]}
    """
    raise NotImplementedError