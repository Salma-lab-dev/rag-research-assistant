# test_crage.py - Test Corrective RAG implementation
from rag.embeddings import load_index
from rag.crage import ask_crage

# Load the FAISS index
db = load_index()
retriever = db.as_retriever(search_kwargs={"k": 4})

# Test CRAG with a query
query = "What is federated learning?"
print(f"Query: {query}\n")

response = ask_crage(query, retriever)

print(f"Answer: {response['answer']}\n")
print(f"Sources:")
for source in response['sources']:
    print(f"  - {source['source']}, page {source['page']}")
    print(f"    Snippet: {source['snippet'][:100]}...")

print(f"\nCRAG Metadata:")
print(f"  Relevance Score: {response['crage_metadata']['relevance_score']}")
print(f"  Refinement Note: {response['crage_metadata']['refinement_note']}")
print(f"  Documents Used: {response['crage_metadata']['docs_used']}")
