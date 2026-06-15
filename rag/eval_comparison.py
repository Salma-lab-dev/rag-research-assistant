# rag/eval_comparison.py - Compare Standard RAG vs CRAG performance
import csv
import os
import ssl
import httpx
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

from rag.ingestion import load_and_chunk
from rag.embeddings import build_index
from rag.retriever import build_qa_chain, ask
from rag.crage import ask_crage

# ── 15 Q&A pairs based on TP7-Federated_learning.pdf ─────────────
TEST_QA = [
    {
        "question": "What is federated learning?",
        "expected_keywords": ["federated", "fédéré", "distribué", "local", "modèle", "clients"]
    },
    {
        "question": "What is horizontal federated learning?",
        "expected_keywords": ["horizontal", "same features", "mêmes features", "clients", "lignes"]
    },
    {
        "question": "What is vertical federated learning?",
        "expected_keywords": ["vertical", "different features", "features différentes", "colonnes", "entités"]
    },
    {
        "question": "What is the FedAvg algorithm?",
        "expected_keywords": ["FedAvg", "average", "moyenne", "agrégation", "poids", "weights"]
    },
    {
        "question": "What is the role of the central server in federated learning?",
        "expected_keywords": ["serveur", "server", "agrégation", "global", "central", "modèle global"]
    },
    {
        "question": "What are the privacy benefits of federated learning?",
        "expected_keywords": ["privacy", "confidential", "données", "local", "privé", "partagé"]
    },
    {
        "question": "How is federated learning applied to medical imaging?",
        "expected_keywords": ["médical", "medical", "hôpital", "hospital", "patient", "images", "CNN"]
    },
    {
        "question": "What is a CNN?",
        "expected_keywords": ["convolution", "CNN", "réseau", "neurones", "couche", "pooling"]
    },
    {
        "question": "What is the difference between IID and non-IID data?",
        "expected_keywords": ["iid", "non-iid", "distribution", "hétérogène", "heterogeneous"]
    },
    {
        "question": "What happens during a local training round?",
        "expected_keywords": ["local", "entraînement", "training", "epoch", "client", "gradient"]
    },
    {
        "question": "What are the main challenges of federated learning?",
        "expected_keywords": ["communication", "défi", "challenge", "heterogeneity", "non-iid", "convergence"]
    },
    {
        "question": "What metrics are used to evaluate the federated model?",
        "expected_keywords": ["accuracy", "précision", "loss", "performance", "évaluation", "metric"]
    },
    {
        "question": "What is model aggregation in federated learning?",
        "expected_keywords": ["agrégation", "aggregation", "moyenne", "average", "poids", "global"]
    },
    {
        "question": "What dataset is used in the practical work?",
        "expected_keywords": ["dataset", "données", "mnist", "médical", "images", "jeu de données"]
    },
    {
        "question": "What is the difference between federated learning and traditional machine learning?",
        "expected_keywords": ["centralisé", "centralized", "distribué", "distributed", "données", "local", "server"]
    },
]


# ── Helpers ───────────────────────────────────────────────────────

def keyword_hit(answer: str, keywords: list) -> bool:
    """Return True if any keyword appears in the answer."""
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)


class FakePDF:
    """Simulates a Streamlit UploadedFile for local testing."""
    def __init__(self, path: str):
        self.name = os.path.basename(path)
        self._path = path

    def read(self):
        with open(self._path, "rb") as f:
            return f.read()


# ── Comparison eval function ───────────────────────────────────────

def run_comparison_eval(test_files):
    print("=" * 70)
    print(" Loading and indexing test files...")
    chunks = load_and_chunk(test_files)
    print(f"   → {len(chunks)} chunks created from {len(test_files)} file(s)")

    db = build_index(chunks)
    retriever = build_qa_chain(db)
    print("   → FAISS index ready\n")
    print("=" * 70)
    print(" Running RAG vs CRAG comparison...\n")

    results = []
    rag_hits = 0
    crage_hits = 0
    crage_relevance_scores = []
    crage_refinement_count = 0

    for i, item in enumerate(TEST_QA, 1):
        # Run standard RAG
        rag_response = ask(item["question"], retriever)
        rag_hit = keyword_hit(rag_response["answer"], item["expected_keywords"])
        rag_hits += int(rag_hit)

        # Run CRAG
        crage_response = ask_crage(item["question"], retriever)
        crage_hit = keyword_hit(crage_response["answer"], item["expected_keywords"])
        crage_hits += int(crage_hit)
        
        # Track CRAG metadata
        relevance_score = crage_response["crage_metadata"]["relevance_score"]
        crage_relevance_scores.append(relevance_score)
        if crage_response["crage_metadata"]["refinement_note"]:
            crage_refinement_count += 1

        # Display comparison
        rag_icon = "✓" if rag_hit else "✗"
        crage_icon = "✓" if crage_hit else "✗"
        
        print(f"[{i:02d}] {item['question'][:60]}")
        print(f"     RAG:  {rag_icon} | CRAG: {crage_icon} | Relevance: {relevance_score:.2f}")
        
        if crage_response["crage_metadata"]["refinement_note"]:
            print(f"     → {crage_response['crage_metadata']['refinement_note']}")
        print()

        results.append({
            "question": item["question"],
            "rag_answer": rag_response["answer"][:200],
            "rag_hit": "YES" if rag_hit else "NO",
            "crage_answer": crage_response["answer"][:200],
            "crage_hit": "YES" if crage_hit else "NO",
            "relevance_score": relevance_score,
            "refinement_used": "YES" if crage_response["crage_metadata"]["refinement_note"] else "NO"
        })

    # ── Summary ───────────────────────────────────────────────────
    rag_hit_rate = rag_hits / len(TEST_QA) * 100
    crage_hit_rate = crage_hits / len(TEST_QA) * 100
    avg_relevance = sum(crage_relevance_scores) / len(crage_relevance_scores)
    
    print("=" * 70)
    print(" COMPARISON RESULTS")
    print("=" * 70)
    print(f" Standard RAG Hit Rate : {rag_hit_rate:.1f}%  ({rag_hits}/{len(TEST_QA)})")
    print(f" CRAG Hit Rate        : {crage_hit_rate:.1f}%  ({crage_hits}/{len(TEST_QA)})")
    print(f" Improvement           : {crage_hit_rate - rag_hit_rate:+.1f}%")
    print()
    print(f" CRAG Average Relevance Score : {avg_relevance:.2f}/1.0")
    print(f" CRAG Refinement Triggered   : {crage_refinement_count}/{len(TEST_QA)} times")
    print("=" * 70)

    # ── Save CSV ──────────────────────────────────────────────────
    csv_path = "eval_comparison_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "rag_answer", "rag_hit", "crage_answer", "crage_hit", "relevance_score", "refinement_used"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f" Results saved to {csv_path}")
    print("=" * 70)
    
    return {
        "rag_hit_rate": rag_hit_rate,
        "crage_hit_rate": crage_hit_rate,
        "avg_relevance": avg_relevance,
        "refinement_count": crage_refinement_count
    }


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    # Add all your test PDFs here
    TEST_PDF_PATHS = [
        r"C:\Users\HP\Downloads\2511.22616v1.pdf",
        # r"C:\Users\HP\Downloads\another_pdf.pdf",  ← add more here
    ]

    # Check all files exist before starting
    missing = [p for p in TEST_PDF_PATHS if not os.path.exists(p)]
    if missing:
        print(" Missing files:")
        for m in missing:
            print(f"   {m}")
        exit(1)

    test_files = [FakePDF(p) for p in TEST_PDF_PATHS]
    run_comparison_eval(test_files)
