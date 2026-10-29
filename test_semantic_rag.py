
from semantic_rag import retrieve_semantic, format_semantic_context

test_messages = [
    (
        "Prize scam",
        "I won a reward but must pay a processing fee "
        "before receiving my prize."
    ),
    (
        "Bank phishing",
        "My bank says my account will be blocked unless "
        "I verify my password using a link."
    ),
    (
        "Credential theft",
        "Someone is asking me to disclose my one-time "
        "password and banking details."
    ),
]

for name, message in test_messages:
    print("\n" + "=" * 55)
    print("TEST:", name)
    print("MESSAGE:", message)

    results = retrieve_semantic(message, top_k=3)

    if not results:
        print("No sufficiently relevant documents retrieved.")
        continue

    for result in results:
        print(
            f"\nTopic: {result['topic']}"
            f"\nSimilarity: {result['similarity']}"
            f"\nSource: {result['source']}"
            f"\nGuidance: {result['content']}"
        )

    print("\n--- CONTEXT FOR OLLAMA ---")
    print(format_semantic_context(results))
