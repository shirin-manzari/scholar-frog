import hashlib
import json
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

from src.ingest import get_collection, get_embedding_model

load_dotenv()

RETRIEVAL_MODES = ("dense", "hybrid", "hybrid-rerank")
_bm25_cache = None
_reranker = None
_reranker_name = None


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str
    dense_candidates: int
    bm25_candidates: int
    rrf_k: int
    rerank_candidates: int
    final_results: int
    reranker_model: str

    @classmethod
    def from_env(cls):
        mode = os.getenv("RETRIEVAL_MODE", "hybrid-rerank").lower()
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"RETRIEVAL_MODE must be one of: {', '.join(RETRIEVAL_MODES)}")
        reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base").strip()
        if not reranker_model:
            raise ValueError("RERANKER_MODEL must not be empty")
        return cls(
            mode=mode,
            dense_candidates=_positive_int("DENSE_CANDIDATES", 20),
            bm25_candidates=_positive_int("BM25_CANDIDATES", 20),
            rrf_k=_positive_int("RRF_K", 60),
            rerank_candidates=_positive_int("RERANK_CANDIDATES", 20),
            final_results=_positive_int("FINAL_RESULTS", 5),
            reranker_model=reranker_model,
        )


def _positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        parsed = int(value) if value is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"(?u)[^\W_][\w.+#/-]*", text.casefold())
    expanded = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(part for part in re.split(r"[.+#/-]+", token) if part != token)
    return expanded


def _corpus_fingerprint(ids: list[str], docs: list[str], metadatas: list[dict]) -> str:
    digest = hashlib.sha256()
    for chunk_id, text, metadata in zip(ids, docs, metadatas):
        digest.update(chunk_id.encode())
        digest.update(b"\0")
        digest.update(text.encode())
        digest.update(b"\0")
        digest.update(json.dumps(metadata or {}, sort_keys=True, default=str).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _get_bm25_index(collection):
    """Reuse BM25 until Chroma's IDs, text, or citation metadata changes."""
    global _bm25_cache

    stored = collection.get(include=["documents", "metadatas"])
    ids = stored["ids"] or []
    docs = stored["documents"] or []
    metadatas = stored["metadatas"] or [{} for _ in ids]
    fingerprint = _corpus_fingerprint(ids, docs, metadatas)
    if _bm25_cache and _bm25_cache[0] == fingerprint:
        return _bm25_cache[1:]

    from rank_bm25 import BM25Okapi

    records = []
    tokenized = []
    for chunk_id, text, metadata in zip(ids, docs, metadatas):
        records.append(_make_result(chunk_id, text, metadata or {}))
        tokenized.append(_tokenize(text))
    index = BM25Okapi(tokenized) if any(tokenized) else None
    _bm25_cache = (fingerprint, index, records, tokenized)
    return index, records, tokenized


def _make_result(chunk_id: str, text: str, metadata: dict) -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "metadata": metadata,
        "source": metadata["source"],
        "title": metadata["title"],
        "section": metadata.get("section", "Untitled section"),
        "page": metadata["page"],
        "distance": None,
        "dense_distance": None,
        "dense_score": None,
        "bm25_score": None,
        "rrf_score": None,
        "reranker_score": None,
    }


def _dense_search(question: str, collection, count: int, candidate_count: int) -> list[dict]:
    model = get_embedding_model()
    query_embedding = model.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(candidate_count, count),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for chunk_id, text, metadata, distance in zip(
        results["ids"][0], results["documents"][0],
        results["metadatas"][0], results["distances"][0],
    ):
        hit = _make_result(chunk_id, text, metadata or {})
        hit["distance"] = hit["dense_distance"] = float(distance)
        hit["dense_score"] = -float(distance)
        hits.append(hit)
    return sorted(hits, key=lambda hit: (hit["distance"], hit["id"]))


def _bm25_search(question: str, collection, candidate_count: int) -> list[dict]:
    query_tokens = _tokenize(question)
    if not query_tokens:
        return []

    index, records, tokenized = _get_bm25_index(collection)
    if index is None:
        return []

    scores = index.get_scores(query_tokens)
    query_terms = set(query_tokens)
    matching = [
        i for i, tokens in enumerate(tokenized)
        if query_terms.intersection(tokens)
    ]
    ranked = sorted(matching, key=lambda i: (-float(scores[i]), records[i]["id"]))
    hits = []
    for i in ranked[:candidate_count]:
        hit = records[i].copy()
        hit["bm25_score"] = float(scores[i])
        hits.append(hit)
    return hits


def reciprocal_rank_fusion(*ranked_lists: list[dict], k: int = 60) -> list[dict]:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    fused = {}
    scores = {}
    for ranked in ranked_lists:
        seen = set()
        unique = []
        for result in ranked:
            chunk_id = result["id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            unique.append(result)
        for rank, result in enumerate(unique, start=1):
            chunk_id = result["id"]
            if chunk_id not in fused:
                fused[chunk_id] = result.copy()
                scores[chunk_id] = 0.0
            else:
                fused[chunk_id].update({
                    key: value for key, value in result.items()
                    if value is not None
                })
            scores[chunk_id] += 1 / (k + rank)

    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    for chunk_id in ordered_ids:
        fused[chunk_id]["rrf_score"] = scores[chunk_id]
    return [fused[chunk_id] for chunk_id in ordered_ids]


def _create_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu", max_length=512)


def _get_reranker(model_name: str):
    global _reranker, _reranker_name
    if _reranker is None or _reranker_name != model_name:
        _reranker = _create_reranker(model_name)
        _reranker_name = model_name
    return _reranker


def _rerank(question: str, candidates: list[dict], model_name: str) -> list[dict]:
    if not candidates:
        return []
    model = _get_reranker(model_name)
    pairs = [(question, candidate["text"]) for candidate in candidates]
    scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
    ranked = sorted(
        zip(candidates, scores),
        key=lambda pair: (-float(pair[1]), -pair[0]["rrf_score"], pair[0]["id"]),
    )
    results = []
    for candidate, score in ranked:
        result = candidate.copy()
        result["reranker_score"] = float(score)
        results.append(result)
    return results


def retrieve(question: str, top_k: int | None = None,
             retrieval_mode: str | None = None) -> list[dict]:
    """Return ranked Chroma chunks with citation metadata and method scores."""
    config = RetrievalConfig.from_env()
    top_k = config.final_results if top_k is None else top_k
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    mode = retrieval_mode or config.mode
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"retrieval_mode must be one of: {', '.join(RETRIEVAL_MODES)}")

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    dense = _dense_search(question, collection, count, config.dense_candidates)
    if mode == "dense":
        return dense[:top_k]

    bm25 = _bm25_search(question, collection, config.bm25_candidates)
    fused = reciprocal_rank_fusion(dense, bm25, k=config.rrf_k)
    if mode == "hybrid":
        return fused[:top_k]

    candidates = fused[:config.rerank_candidates]
    return _rerank(question, candidates, config.reranker_model)[:top_k]
