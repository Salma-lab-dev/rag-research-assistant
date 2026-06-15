# rag/crage.py - Corrective RAG Implementation
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

chat_history = []

def get_llm():
    client = httpx.Client(verify=False)
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        http_client=client,
    )

def evaluate_relevance(query: str, docs) -> dict:
    """
    Evaluate the relevance of retrieved documents to the query.
    Returns a score (0-1) and assessment.
    """
    docs_text = "\n\n".join([f"Doc {i+1}: {doc.page_content[:300]}" for i, doc in enumerate(docs)])
    
    eval_prompt = PromptTemplate(
        input_variables=["query", "docs"],
        template="""Rate the relevance of these documents to the query on a scale of 0.0 to 1.0.
Query: {query}

Documents:
{docs}

Provide your rating as a single number (e.g., 0.85) followed by a brief explanation.
Rating:"""
    )
    
    llm = get_llm()
    formatted = eval_prompt.format(query=query, docs=docs_text)
    response = llm.invoke(formatted)
    
    # Parse the score from response
    try:
        score = float(response.content.strip().split()[0])
    except:
        score = 0.5  # default if parsing fails
    
    return {
        "score": score,
        "explanation": response.content,
        "is_relevant": score >= 0.3
    }

def refine_knowledge(query: str, docs) -> str:
    """
    Refine retrieved documents by extracting and synthesizing relevant information.
    """
    docs_text = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    
    refine_prompt = PromptTemplate(
        input_variables=["query", "docs"],
        template="""Extract and synthesize the most relevant information from these documents to answer the query.
Focus on facts, figures, and direct answers. Remove irrelevant content.

Query: {query}

Documents:
{docs}

Refined knowledge:"""
    )
    
    llm = get_llm()
    formatted = refine_prompt.format(query=query, docs=docs_text)
    response = llm.invoke(formatted)
    
    return response.content

def search_additional(query: str, retriever, max_attempts: int = 2) -> list:
    """
    Perform additional retrieval attempts with query variations when initial docs are not relevant.
    """
    variations = [
        query,
        f"key concepts about {query}",
        f"important information regarding {query}",
        f"details about {query}"
    ]
    
    all_docs = []
    for i, variation in enumerate(variations[:max_attempts]):
        docs = retriever.invoke(variation)
        all_docs.extend(docs)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_docs = []
    for doc in all_docs:
        doc_hash = hash(doc.page_content[:100])
        if doc_hash not in seen:
            seen.add(doc_hash)
            unique_docs.append(doc)
    
    return unique_docs[:6]  # Return top 6 unique docs

def ask_crage(query: str, retriever) -> dict:
    """
    Corrective RAG (CRAG) implementation with knowledge evaluation, refinement, and searching.
    """
    # Step 1: Initial retrieval
    docs = retriever.invoke(query)
    
    # Step 2: Evaluate relevance
    evaluation = evaluate_relevance(query, docs)
    
    refined_docs = docs
    refinement_note = ""
    
    # Step 3: If not relevant, search with variations
    if not evaluation["is_relevant"]:
        refinement_note = "Initial documents had low relevance. Performing additional search..."
        refined_docs = search_additional(query, retriever)
        
        # Re-evaluate after additional search
        new_evaluation = evaluate_relevance(query, refined_docs)
        evaluation = new_evaluation
    
    # Step 4: Refine knowledge
    refined_knowledge = refine_knowledge(query, refined_docs)
    
    # Step 5: Generate answer with refined knowledge
    crag_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful research assistant.
Answer the question using the refined knowledge below.
For every fact you state, cite the source document name and page number like: (source: filename.pdf, page 3).
If the answer is not in the context, say "I don't have enough information in the uploaded documents."

Refined Knowledge:
{context}

Question: {question}
Answer:"""
    )
    
    formatted = crag_prompt.format(context=refined_knowledge, question=query)
    llm = get_llm()
    response = llm.invoke(formatted)
    answer = response.content
    
    chat_history.append({"user": query, "assistant": answer})
    if len(chat_history) > 5:
        chat_history.pop(0)
    
    sources = []
    for doc in refined_docs:
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:200]
        })
    
    return {
        "answer": answer,
        "sources": sources,
        "crage_metadata": {
            "relevance_score": evaluation["score"],
            "refinement_note": refinement_note,
            "docs_used": len(refined_docs)
        }
    }
