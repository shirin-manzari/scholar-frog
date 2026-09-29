import hashlib
import json
import math
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

from src.ingest import get_collection, get_embedding_model, get_index_config

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
    reranker_min_score: float
    max_chunks_per_paper: int
    mmr_lambda: float
    adjacent_chunks: int

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
            reranker_min_score=_finite_float("RERANKER_MIN_SCORE", 0.05),
            max_chunks_per_paper=_positive_int("MAX_CHUNKS_PER_PAPER", 2),
            mmr_lambda=_bounded_float("MMR_LAMBDA", 0.75, 0.0, 1.0),
            adjacent_chunks=_non_negative_int("ADJACENT_CHUNKS", 1),
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


def _non_negative_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        parsed = int(value) if value is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _finite_float(name: str, default: float) -> float:
    value = os.getenv(name)
    try:
        parsed = float(value) if value is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    parsed = _finite_float(name, default)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
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


def _get_committed_snapshot(collection):
    from src.sync import committed_snapshot

    return committed_snapshot(collection_count=collection.count())


def _get_bm25_index(collection, snapshot):
    """Build BM25 only from committed IDs; cache by the committed manifest revision."""
    global _bm25_cache

    collection_key = getattr(collection, "name", id(collection))
    cache_key = (collection_key, snapshot.revision)
    if _bm25_cache and _bm25_cache[0] == cache_key:
        return _bm25_cache[1:]

    if not snapshot.chunk_ids:
        _bm25_cache = (cache_key, None, [], [])
        return _bm25_cache[1:]

    ids = sorted(snapshot.chunk_ids)
    stored = collection.get(ids=ids, include=["documents", "metadatas"])
    ids = stored["ids"] or []
    docs = stored["documents"] or []
    metadatas = stored["metadatas"] or [{} for _ in ids]

    from rank_bm25 import BM25Okapi

    records = []
    tokenized = []
    for chunk_id, text, metadata in zip(ids, docs, metadatas):
        owner = snapshot.owners.get(chunk_id)
        metadata = metadata or {}
        if owner is None or (metadata.get("document_id") and metadata["document_id"] != owner["document_id"]):
            continue
        records.append(_make_committed_result(chunk_id, text, metadata, owner))
        tokenized.append(_tokenize(text))
    index = BM25Okapi(tokenized) if any(tokenized) else None
    _bm25_cache = (cache_key, index, records, tokenized)
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


def _make_committed_result(chunk_id: str, text: str, metadata: dict, owner: dict) -> dict:
    metadata = dict(metadata)
    paths = owner.get("paths") or []
    if paths:
        metadata["source"] = paths[0]
        metadata["source_paths"] = json.dumps(paths)
    metadata["document_id"] = owner["document_id"]
    result = _make_result(chunk_id, text, metadata)
    return result


def _dense_search(question: str, collection, count: int, candidate_count: int,
                  snapshot) -> list[dict]:
    if not snapshot.chunk_ids or candidate_count <= 0:
        return []
    config = get_index_config()
    model = get_embedding_model()
    query_embedding = model.encode(
        [question], normalize_embeddings=config.normalize_embeddings
    ).tolist()
    if len(query_embedding[0]) != config.embedding_dimension:
        raise ValueError(
            f"Query embedding dimension {len(query_embedding[0])} does not match "
            f"index dimension {config.embedding_dimension}; rebuild or correct configuration."
        )
    target = min(candidate_count, len(snapshot.chunk_ids))
    fetch = min(max(target, 1), count)
    hits = []
    while fetch:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=fetch,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0],
        ):
            owner = snapshot.owners.get(chunk_id)
            metadata = metadata or {}
            if owner is None or (metadata.get("document_id") and metadata["document_id"] != owner["document_id"]):
                continue
            hit = _make_committed_result(chunk_id, text, metadata, owner)
            hit["distance"] = hit["dense_distance"] = float(distance)
            hit["dense_score"] = -float(distance)
            hits.append(hit)
        if len(hits) >= target or fetch >= count:
            break
        fetch = min(count, max(fetch + 1, fetch * 2))
    return sorted(hits, key=lambda hit: (hit["distance"], hit["id"]))


def _bm25_search(question: str, collection, candidate_count: int, snapshot) -> list[dict]:
    query_tokens = _tokenize(question)
    if not query_tokens:
        return []

    index, records, tokenized = _get_bm25_index(collection, snapshot)
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


def _paper_key(result: dict) -> str:
    metadata = result.get("metadata") or {}
    return str(
        metadata.get("document_id")
        or metadata.get("file_hash")
        or result.get("source")
        or result.get("id")
    )


def _text_similarity(left: str, right: str) -> float:
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _mmr_select(candidates: list[dict], limit: int, diversity: float,
                max_per_paper: int) -> list[dict]:
    """Select relevant but non-redundant anchors using lexical MMR."""
    if not candidates or limit <= 0:
        return []
    raw_scores = [float(candidate["reranker_score"]) for candidate in candidates]
    low, high = min(raw_scores), max(raw_scores)

    def relevance(candidate):
        if high == low:
            return 1.0
        return (float(candidate["reranker_score"]) - low) / (high - low)

    selected = []
    per_paper = {}
    remaining = list(candidates)
    while remaining and len(selected) < limit:
        eligible = [
            candidate for candidate in remaining
            if per_paper.get(_paper_key(candidate), 0) < max_per_paper
        ]
        if not eligible:
            break
        scored = []
        for candidate in eligible:
            redundancy = max(
                (_text_similarity(candidate["text"], chosen["text"])
                 for chosen in selected),
                default=0.0,
            )
            mmr_score = diversity * relevance(candidate) - (1 - diversity) * redundancy
            scored.append((mmr_score, float(candidate["reranker_score"]), candidate))
        _, _, chosen = sorted(
            scored, key=lambda item: (-item[0], -item[1], item[2]["id"])
        )[0]
        selected.append(chosen)
        key = _paper_key(chosen)
        per_paper[key] = per_paper.get(key, 0) + 1
        remaining.remove(chosen)
    return selected


_CHUNK_POSITION = re.compile(
    r"^[0-9a-f]{16,64}-(?P<page>\d+)-(?P<index>\d+)(?:-[0-9a-f]{16})?$",
    re.IGNORECASE,
)


def _chunk_position(result: dict) -> tuple[int, int] | None:
    metadata = result.get("metadata") or {}
    page = metadata.get("page", result.get("page"))
    index = metadata.get("chunk_index")
    try:
        page = int(page)
        if index is not None:
            return page, int(index)
    except (TypeError, ValueError):
        return None
    match = _CHUNK_POSITION.match(str(result.get("id", "")))
    if not match or int(match.group("page")) != page:
        return None
    return page, int(match.group("index"))


def _select_context(reranked: list[dict], corpus: list[dict], *, top_k: int,
                    min_score: float, max_per_paper: int, diversity: float,
                    adjacent_chunks: int) -> list[dict]:
    """Threshold, diversify, cap, and expand reranked evidence."""
    relevant = [
        candidate for candidate in reranked
        if float(candidate["reranker_score"]) >= min_score
    ]
    if not relevant:
        return []

    anchors = _mmr_select(relevant, top_k, diversity, max_per_paper)
    if not anchors:
        return []
    if adjacent_chunks == 0:
        return [{**anchor, "selection_reason": "anchor", "adjacent_to": None}
                for anchor in anchors]

    by_paper = {}
    for record in corpus:
        position = _chunk_position(record)
        if position is not None:
            by_paper.setdefault(_paper_key(record), []).append((position, record))
    for records in by_paper.values():
        records.sort(key=lambda item: (item[0], item[1]["id"]))

    ranked_by_id = {candidate["id"]: candidate for candidate in reranked}
    selected = []
    selected_ids = set()
    per_paper = {}

    def add(result: dict, reason: str, adjacent_to: str | None = None) -> bool:
        key = _paper_key(result)
        if (result["id"] in selected_ids or len(selected) >= top_k
                or per_paper.get(key, 0) >= max_per_paper):
            return False
        selected.append({
            **result,
            "selection_reason": reason,
            "adjacent_to": adjacent_to,
        })
        selected_ids.add(result["id"])
        per_paper[key] = per_paper.get(key, 0) + 1
        return True

    group_size = 1 + (2 * adjacent_chunks)
    initial_anchor_count = min(len(anchors), max(1, math.ceil(top_k / group_size)))
    for anchor in anchors[:initial_anchor_count]:
        if not add(anchor, "anchor"):
            continue
        ordered = by_paper.get(_paper_key(anchor), [])
        anchor_index = next(
            (i for i, (_, record) in enumerate(ordered) if record["id"] == anchor["id"]),
            None,
        )
        if anchor_index is None:
            continue
        for distance in range(1, adjacent_chunks + 1):
            # Later chunks are preferred because overlap already carries the
            # preceding chunk's tail into the anchor in the current chunker.
            for neighbor_index in (anchor_index + distance, anchor_index - distance):
                if 0 <= neighbor_index < len(ordered):
                    neighbor = ordered[neighbor_index][1]
                    neighbor = ranked_by_id.get(neighbor["id"], neighbor)
                    add(neighbor, "adjacent", anchor["id"])

    # If boundaries or paper caps left reserved neighbor slots unused, fill
    # them with the remaining diverse, threshold-passing anchors.
    for anchor in anchors[initial_anchor_count:]:
        add(anchor, "anchor")
    return selected


def _retrieve_locked(question: str, top_k: int | None = None,
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

    snapshot = _get_committed_snapshot(collection)
    if not snapshot.chunk_ids:
        return []

    dense = _dense_search(question, collection, count, config.dense_candidates, snapshot)
    if mode == "dense":
        return dense[:top_k]

    bm25 = _bm25_search(question, collection, config.bm25_candidates, snapshot)
    fused = reciprocal_rank_fusion(dense, bm25, k=config.rrf_k)
    if mode == "hybrid":
        return fused[:top_k]

    candidates = fused[:config.rerank_candidates]
    reranked = _rerank(question, candidates, config.reranker_model)
    _, corpus, _ = _get_bm25_index(collection, snapshot)
    return _select_context(
        reranked,
        corpus,
        top_k=top_k,
        min_score=config.reranker_min_score,
        max_per_paper=config.max_chunks_per_paper,
        diversity=config.mmr_lambda,
        adjacent_chunks=config.adjacent_chunks,
    )


def retrieve(question: str, top_k: int | None = None,
             retrieval_mode: str | None = None) -> list[dict]:
    """Search a consistent committed manifest snapshot in every retrieval mode."""
    from src.sync import INDEX_LOCK

    with INDEX_LOCK:
        return _retrieve_locked(question, top_k=top_k, retrieval_mode=retrieval_mode)
