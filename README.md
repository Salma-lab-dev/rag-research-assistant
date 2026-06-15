
# 📄 RAG Research Assistant

A conversational research assistant that lets you upload PDF documents and ask questions about them. Powered by LangChain, FAISS, local sentence-transformers embeddings, and Groq API (Llama 3.3 70B).


> Built as part of the ENSIAS 2IA — Apprentissage Neuronal Profond · NLP Project

---

## 🔗 Live demo
👉 https://rag-research-assistant-6natfewtcr3awabbu3o5al.streamlit.app/

---

## ✨ Features
- Upload multiple PDFs and build a searchable knowledge base
- Ask natural language questions about your documents
- Answers include inline citations (document name + page number)
- Conversation memory across multiple turns
- **Corrective RAG (CRAG)**: Advanced retrieval with knowledge evaluation, refinement, and searching for improved accuracy
- Clean, responsive UI built with Streamlit

---

## 🗂️ Project structure
```
rag-research-assistant/
├── app.py                  → Streamlit entry point (Person B)
├── ui/
│   └── components.py       → Reusable UI widgets (Person B)
├── rag/
│   ├── ingestion.py        → PDF loading + chunking (Person A)
│   ├── embeddings.py       → Embedding + FAISS index (Person A)
│   ├── retriever.py        → Standard RAG retrieval (Person A)
│   ├── crage.py           → Corrective RAG implementation (Person A)
│   └── eval.py             → Benchmark evaluation (Person A)
├── requirements.txt
└── README.md
```

---

## ⚙️ Architecture
| Component | Tool |
|---|---|
| PDF Ingestion | PyMuPDF / PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter (chunk=800, overlap=100) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (local, free) |
| Vector Store | FAISS (persisted to disk) |
| Retrieval | Similarity search, Top-K = 4 |
| CRAG | Knowledge evaluation, refinement, and searching |
| LLM | Groq API — Llama 3 (free tier) |
| Memory | ConversationBufferWindowMemory k=5 |
| UI | Streamlit |
| Deployment | Streamlit Cloud |

---

## 🧠 Corrective RAG (CRAG)

CRAG enhances traditional RAG by adding three key steps:

1. **Knowledge Evaluation**: Assesses the relevance score (0-1) of retrieved documents using the LLM
2. **Knowledge Refinement**: If documents are relevant, extracts and synthesizes key information to remove noise
3. **Knowledge Searching**: If relevance is low (< 0.6), performs additional retrieval with query variations to find better matches

This reduces hallucinations and improves answer accuracy by ensuring only high-quality, relevant context is used for generation.

**Usage:**
```python
from rag.embeddings import load_index
from rag.crage import ask_crage

db = load_index()
retriever = db.as_retriever(search_kwargs={"k": 4})
response = ask_crage("What is federated learning?", retriever)

print(response['answer'])
print(f"Relevance Score: {response['crage_metadata']['relevance_score']}")
```

---

## Run locally
```bash
git clone https://github.com/Salma-lab-dev/rag-research-assistant
cd rag-research-assistant
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Eval results
============================================================
 Hit Rate : 93.3%  (14/15)
 Target reached (≥ 85%) — retrieval quality is good!
 Hit Rate : 93.3%  (14/15)
 Target reached (≥ 85%) — retrieval quality is good!
============================================================

## 👥 Team
Hajar EL HALLAGUE & Salma KAMAL
```
