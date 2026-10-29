
from hybrid_rag import retrieve_hybrid, format_hybrid_context

tests = [
    (
        "Prize scam",
        "I won a reward but must pay a processing fee before receiving it."
    ),
    (
        "Bank phishing",
        "My bank says my account will be blocked unless I verify my password."
    ),
    (
        "Credential theft",
        "Someone asks me for my one-time password and banking details."
    ),
]

for name, message in tests:
    print("\n" + "=" * 60)
    print("TEST:", name)
    print("MESSAGE:", message)

    results = retrieve_hybrid(message, top_k=3)

    if not results:
        print("No relevant guidance retrieved.")
        continue

    for doc in results:
        print(f"\nTopic: {doc['topic']}")
        print(f"Source: {doc.get('source', 'Not specified')}")
        print(f"RRF score: {doc['rrf_score']}")
        print(f"Guidance: {doc['content']}")

    print("\n--- CONTEXT FOR OLLAMA ---")
    print(format_hybrid_context(results))
