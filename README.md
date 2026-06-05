
# 📄 RAG Research Assistant

A conversational research assistant that lets you upload PDF documents and ask questions about them. Powered by LangChain, FAISS, and Groq (Llama 3).

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
│   ├── retriever.py        → RetrievalQA chain (Person A)
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
| LLM | Groq API — Llama 3 (free tier) |
| Memory | ConversationBufferWindowMemory k=5 |
| UI | Streamlit |
| Deployment | Streamlit Cloud |

---

## 🚀 Run locally
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
_To be added by Person A — Day 5_

---

## 👥 Team
| Person | Role |
|---|---|
| Person B (Salma) | Streamlit UI · Integration · Deployment |
| Person A | RAG pipeline · Embeddings · Retrieval · Evaluation |
```
