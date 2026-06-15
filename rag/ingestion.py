# rag/ingestion.py
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile, os

def load_and_chunk(uploaded_files):
    """
    Accepts a list of Streamlit UploadedFile objects.
    Returns a list of LangChain Document chunks.
    """
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    for uploaded_file in uploaded_files:
        # Save to temp file because PyMuPDFLoader needs a path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()

        # Tag each chunk with the original filename
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
        os.unlink(tmp_path)  # delete temp file

    return all_chunks