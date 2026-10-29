
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = (
    BASE_DIR / "knowledge_base" / "cybersecurity_guidance.json"
)


def load_knowledge_base():
    """Load cybersecurity guidance from the local JSON file."""

    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_FILE}"
        )

    with KNOWLEDGE_FILE.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list) or not documents:
        raise ValueError("The knowledge base is empty or invalid.")

    return documents


def retrieve_guidance(message, top_k=3):
    """Retrieve the most relevant knowledge-base entries."""

    documents = load_knowledge_base()

    texts = [
        f"{doc['topic']} {doc['content']}"
        for doc in documents
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )

    document_vectors = vectorizer.fit_transform(texts)
    message_vector = vectorizer.transform([message])

    similarities = cosine_similarity(
        message_vector,
        document_vectors,
    ).flatten()

    # If there are no matching vocabulary terms, don't invent relevance.
    if message_vector.nnz == 0 or similarities.max() <= 0:
        return []

    ranked_indices = similarities.argsort()[::-1]
    results = []

    for index in ranked_indices:
        score = float(similarities[index])

        if score <= 0:
            continue

        document = documents[index]

        results.append({
            "id": document["id"],
            "topic": document["topic"],
            "source": document["source"],
            "url": document["url"],
            "content": document["content"],
            "similarity": round(score, 3),
        })

        if len(results) >= top_k:
            break

    return results


def format_guidance_for_llm(retrieved_documents):
    """Format retrieved documents as context for the local LLM."""

    if not retrieved_documents:
        return (
            "No sufficiently relevant guidance was found in the "
            "local knowledge base. Do not claim that a source "
            "supports a conclusion."
        )

    sections = []

    for doc in retrieved_documents:
        sections.append(
            f"Topic: {doc['topic']}\n"
            f"Source: {doc['source']}\n"
            f"Reference URL: {doc['url']}\n"
            f"Guidance: {doc['content']}"
        )

    return "\n\n".join(sections)
