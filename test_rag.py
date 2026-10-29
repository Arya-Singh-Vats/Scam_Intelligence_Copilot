
from rag_engine import retrieve_guidance, format_guidance_for_llm

message = (
    "Congratulations! You have won a cash prize. "
    "Claim your prize at www.example.com and pay a processing fee."
)

results = retrieve_guidance(message)

print("\nRetrieved documents:", len(results))

for doc in results:
    print("\nTopic:", doc["topic"])
    print("Source:", doc["source"])
    print("Similarity:", doc["similarity"])
    print("Guidance:", doc["content"])
    print("Reference:", doc["url"])

print("\n--- CONTEXT FOR OLLAMA ---")
print(format_guidance_for_llm(results))
