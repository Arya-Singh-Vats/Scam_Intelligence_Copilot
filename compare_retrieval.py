
from rag_engine import retrieve_guidance
from semantic_rag import retrieve_semantic

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
    print("\n" + "=" * 65)
    print("TEST:", name)
    print("MESSAGE:", message)

    print("\n--- TF-IDF RETRIEVAL ---")
    for doc in retrieve_guidance(message, top_k=3):
        print(
            f"{doc['topic']} | "
            f"score={doc['similarity']}"
        )

    print("\n--- SEMANTIC RETRIEVAL ---")
    for doc in retrieve_semantic(message, top_k=3):
        print(
            f"{doc['topic']} | "
            f"score={doc['similarity']}"
        )
