from src.ingest import get_collection, get_embedding_model


def retrieve(question: str, top_k: int = 6) -> list[dict]:
    """Returns a list of {text, source, title, page, distance} dicts,
    most relevant first."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    model = get_embedding_model()

    query_embedding = model.encode([question], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, count),
    )

    hits = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({
            "text": doc,
            "source": meta["source"],
            "title": meta["title"],
            "page": meta["page"],
            "distance": dist,
        })
    return hits
