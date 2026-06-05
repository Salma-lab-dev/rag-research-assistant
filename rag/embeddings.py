# rag/embeddings.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_PATH = "index/"
MODEL_PATH = "models/all-MiniLM-L6-v2"   # local folder

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={"local_files_only": True}
    )

def build_index(chunks):
    emb = get_embeddings()
    db = FAISS.from_documents(chunks, emb)
    db.save_local(INDEX_PATH)
    return db

def load_index():
    emb = get_embeddings()
    db = FAISS.load_local(
        INDEX_PATH, emb,
        allow_dangerous_deserialization=True
    )
    return db