# rag/retriever.py
import os
import ssl
import httpx
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from rag.embeddings import load_index

# ── SSL patch (must happen before any Groq import) ──────────────────
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import groq as _groq_module
_groq_module.DefaultHttpxClient = lambda **kwargs: httpx.Client(verify=False)

from langchain_groq import ChatGroq

load_dotenv()

SYSTEM_PROMPT = """You are a helpful research assistant.
Answer the question using ONLY the provided context.
For every fact you state, cite the source document name and page number like: (source: filename.pdf, page 3).
If the answer is not in the context, say "I don't have enough information in the uploaded documents."

Context:
{context}

Question: {question}
Answer:"""

chat_history = []

def get_llm():
    client = httpx.Client(verify=False)
    return ChatGroq(
        model="llama-3.3-70b-versatile",  # updated model
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        http_client=client,
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_qa_chain(db=None):
    if db is None:
        db = load_index()
    return db.as_retriever(search_kwargs={"k": 4})

def ask(query: str, retriever) -> dict:
    docs = retriever.invoke(query)
    context = format_docs(docs)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=SYSTEM_PROMPT
    )
    formatted = prompt.format(context=context, question=query)

    llm = get_llm()
    response = llm.invoke(formatted)
    answer = response.content

    chat_history.append({"user": query, "assistant": answer})
    if len(chat_history) > 5:
        chat_history.pop(0)

    sources = []
    for doc in docs:
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:200]
        })

    return {"answer": answer, "sources": sources}