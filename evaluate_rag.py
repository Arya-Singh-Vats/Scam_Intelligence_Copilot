
import json
from pathlib import Path

from rag_engine import retrieve_guidance
from semantic_rag import retrieve_semantic
from hybrid_rag import retrieve_hybrid


BASE_DIR = Path(__file__).resolve().parent
TEST_FILE = BASE_DIR / "rag_test_cases.json"


def recall_at_k(retrieved_ids, relevant_ids, k=3):
    """Fraction of relevant documents found in the top k results."""
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    found = set(retrieved_ids[:k]) & relevant
    return len(found) / len(relevant)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """Reciprocal rank of the first relevant result."""
    relevant = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def evaluate_method(test_cases, retrieval_function, method_name):
    recalls = []
    reciprocal_ranks = []
    details = []

    for case in test_cases:
        results = retrieval_function(case["message"], top_k=3)
        retrieved_ids = [doc["id"] for doc in results]
        relevant_ids = case["relevant_doc_ids"]

        recall = recall_at_k(retrieved_ids, relevant_ids, k=3)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)

        recalls.append(recall)
        reciprocal_ranks.append(rr)

        details.append({
            "category": case["category"],
            "expected": relevant_ids,
            "retrieved": retrieved_ids,
            "recall_at_3": recall,
            "reciprocal_rank": rr,
        })

    return {
        "method": method_name,
        "recall_at_3": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "details": details,
    }


def main():
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_FILE}"
        )

    with TEST_FILE.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    if not test_cases:
        raise ValueError("The test dataset is empty.")

    methods = [
        ("TF-IDF", retrieve_guidance),
        ("Semantic", retrieve_semantic),
        ("Hybrid", retrieve_hybrid),
    ]

    reports = []

    for method_name, function in methods:
        print(f"\nEvaluating {method_name} retrieval...")
        report = evaluate_method(
            test_cases,
            function,
            method_name,
        )
        reports.append(report)

    print("\n" + "=" * 68)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 68)

    for report in reports:
        print(f"\nMethod: {report['method']}")
        print(f"Recall@3: {report['recall_at_3']:.3f}")
        print(f"MRR:      {report['mrr']:.3f}")

        for item in report["details"]:
            print(f"\n  Test: {item['category']}")
            print(f"  Expected IDs:  {item['expected']}")
            print(f"  Retrieved IDs: {item['retrieved']}")
            print(f"  Recall@3: {item['recall_at_3']:.3f}")
            print(f"  Reciprocal rank: {item['reciprocal_rank']:.3f}")

    best_recall = max(reports, key=lambda r: r["recall_at_3"])
    best_mrr = max(reports, key=lambda r: r["mrr"])

    print("\n" + "-" * 68)
    print(f"Highest Recall@3: {best_recall['method']}")
    print(f"Highest MRR:      {best_mrr['method']}")
    print(
        "\nNote: these scores measure agreement with your initial "
        "test labels, not overall scam-detection accuracy."
    )


if __name__ == "__main__":
    main()
