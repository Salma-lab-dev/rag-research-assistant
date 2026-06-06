# rag/eval.py
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


# ── Main eval function ────────────────────────────────────────────

def run_eval(test_files):
    print("=" * 60)
    print(" Loading and indexing test files...")
    chunks = load_and_chunk(test_files)
    print(f"   → {len(chunks)} chunks created from {len(test_files)} file(s)")

    db = build_index(chunks)
    retriever = build_qa_chain(db)
    print("   → FAISS index ready\n")
    print("=" * 60)
    print(" Running evaluation...\n")

    results = []
    hits = 0

    for i, item in enumerate(TEST_QA, 1):
        response = ask(item["question"], retriever)
        hit = keyword_hit(response["answer"], item["expected_keywords"])
        hits += int(hit)

        icon = "done" if hit else "error"
        print(f"{icon} [{i:02d}] {item['question'][:65]}")

        results.append({
            "question": item["question"],
            "answer": response["answer"][:300],
            "hit": "YES" if hit else "NO",
            "sources": ", ".join(
                f"{s['source']} p.{s['page']}" for s in response["sources"]
            )
        })

    # ── Summary ───────────────────────────────────────────────────
    hit_rate = hits / len(TEST_QA) * 100
    print("\n" + "=" * 60)
    print(f" Hit Rate : {hit_rate:.1f}%  ({hits}/{len(TEST_QA)})")

    if hit_rate >= 85:
        print(" Target reached (≥ 85%) — retrieval quality is good!")
    else:
        print("  Below 85% — consider enabling EnsembleRetriever.")

    # ── Save CSV ──────────────────────────────────────────────────
    csv_path = "eval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "answer", "hit", "sources"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f" Results saved to {csv_path}")
    print("=" * 60)
    return hit_rate


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
    run_eval(test_files)