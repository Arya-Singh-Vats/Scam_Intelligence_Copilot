
from rag_engine import retrieve_guidance
from semantic_rag import retrieve_semantic


# Terms that indicate a message is genuinely about a delivery.
DELIVERY_TERMS = {
    "delivery", "delivered", "parcel", "package", "shipment",
    "shipping", "courier", "usps", "postage", "tracking",
    "customs", "delivery fee", "redelivery"
}


def reciprocal_rank_fusion(*rankings, k=60):
    """
    Combine ranked document lists using Reciprocal Rank Fusion.
    Documents ranked highly by both retrieval methods receive
    stronger scores.
    """
    fused_scores = {}
    documents_by_id = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            doc_id = doc["id"]

            fused_scores[doc_id] = (
                fused_scores.get(doc_id, 0.0)
                + 1.0 / (k + rank)
            )

            documents_by_id[doc_id] = doc

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    results = []

    for doc_id in ranked_ids:
        doc = documents_by_id[doc_id].copy()
        doc["rrf_score"] = round(fused_scores[doc_id], 5)
        results.append(doc)

    return results


def _contains_delivery_intent(message):
    """Check whether the message specifically concerns delivery."""
    text = message.lower()

    return any(term in text for term in DELIVERY_TERMS)


def _is_irrelevant_delivery_document(doc, message):
    """
    Prevent delivery-specific guidance from being retrieved for
    unrelated messages, such as bank-password phishing messages.
    """
    metadata = " ".join([
        str(doc.get("topic", "")),
        str(doc.get("title", "")),
    ]).lower()

    delivery_document = any(
        term in metadata
        for term in (
            "delivery", "parcel", "usps", "shipping",
            "shipment", "package", "courier"
        )
    )

    return (
        delivery_document
        and not _contains_delivery_intent(message)
    )


def retrieve_hybrid(message, top_k=3):
    """
    Retrieve guidance using keyword and semantic search,
    combine the rankings, and remove clearly irrelevant
    delivery-specific documents.
    """
    if not message or not message.strip():
        return []

    keyword_results = retrieve_guidance(
        message, top_k=top_k
    )
    semantic_results = retrieve_semantic(
        message, top_k=top_k
    )

    combined = reciprocal_rank_fusion(
        keyword_results,
        semantic_results,
    )

    # Filter before selecting the final top-k results.
    relevant_results = [
        doc for doc in combined
        if not _is_irrelevant_delivery_document(doc, message)
    ]

    return relevant_results[:top_k]


def format_hybrid_context(results):
    """Prepare retrieved guidance for the local LLM."""
    if not results:
        return (
            "No relevant cybersecurity guidance was retrieved. "
            "Do not invent sources or claim that guidance was found."
        )

    sections = []

    for doc in results:
        sections.append(
            f"Topic: {doc['topic']}\n"
            f"Source: {doc.get('source', 'Not specified')}\n"
            f"Title: {doc.get('title', doc['topic'])}\n"
            f"Reference: {doc.get('url', '')}\n"
            f"Guidance: {doc['content']}\n"
            f"RRF ranking score: {doc['rrf_score']}"
        )

    return "\n\n".join(sections)
