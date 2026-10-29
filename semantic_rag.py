
from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = (
    BASE_DIR / "knowledge_base" / "cybersecurity_guidance.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load the embedding model once per Python process.
_model = None
_documents = None
_document_embeddings = None


def _load_resources():
    """Load the knowledge base and create document embeddings."""
    global _model, _documents, _document_embeddings

    if _model is not None:
        return

    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_FILE}"
        )

    with KNOWLEDGE_FILE.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list) or not documents:
        raise ValueError("The knowledge base is empty or invalid.")

    for doc in documents:
        if not doc.get("topic") or not doc.get("content"):
            raise ValueError(
                "Every document must contain topic and content."
            )

    model = SentenceTransformer(MODEL_NAME)

    document_texts = [
        f"Topic: {doc['topic']}. Guidance: {doc['content']}"
        for doc in documents
    ]

    embeddings = model.encode(
        document_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    _model = model
    _documents = documents
    _document_embeddings = embeddings


def retrieve_semantic(message, top_k=3):
    """Return the most semantically relevant guidance documents."""
    if not message or not message.strip():
        return []

    if top_k < 1:
        return []

    _load_resources()

    query_embedding = _model.encode(
        [message],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # Normalized vectors: dot product equals cosine similarity.
    scores = np.dot(_document_embeddings, query_embedding)

    ranked_indices = np.argsort(scores)[::-1]
    results = []

    for index in ranked_indices:
        score = float(scores[index])

        # Avoid returning unrelated documents just to fill top_k.
        # This is a provisional threshold, to be evaluated on test queries.
        if score < 0.25:
            continue

        doc = _documents[int(index)]

        results.append({
            "id": doc.get("id", "Unknown"),
            "topic": doc["topic"],
            "source": doc.get("source", "Source not specified"),
            "title": doc.get("title", doc["topic"]),
            "url": doc.get("url", ""),
            "content": doc["content"],
            "similarity": round(score, 3),
        })

        if len(results) >= top_k:
            break

    return results


def format_semantic_context(results):
    """Format retrieved documents for an LLM prompt."""
    if not results:
        return (
            "No sufficiently relevant document was retrieved. "
            "Do not invent supporting sources."
        )

    sections = []

    for doc in results:
        sections.append(
            f"Topic: {doc['topic']}\n"
            f"Source: {doc['source']}\n"
            f"Title: {doc['title']}\n"
            f"Reference: {doc['url']}\n"
            f"Guidance: {doc['content']}"
        )

    return "\n\n".join(sections)
