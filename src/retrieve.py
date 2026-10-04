import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher, get_close_matches

from dotenv import load_dotenv

from src.evidence_policy import RetrievalPlan, infer_plan, select_anchors
from src.relevance import score_contract, effective_threshold

from src.ingest import get_collection, get_embedding_model, get_index_config

load_dotenv()

RETRIEVAL_MODES = ("dense", "hybrid", "hybrid-rerank")
_bm25_cache = None
_reranker = None
_reranker_name = None
_terminology_cache = None
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class QueryTokenLimitError(ValueError):
    """An original query must fail explicitly; optional variants may be rejected."""


def _dense_query_text(question, model, config):
    from src.ingest import token_count, tokenizer_limits
    encoded = query_instruction(config.embedding_model) + question
    tokenizer, capacity = tokenizer_limits(model)
    required = token_count(encoded, tokenizer)
    if required > capacity:
        raise QueryTokenLimitError(
            f"Dense query requires {required} tokens including its query instruction; "
            f"embedding capacity is {capacity}. Shorten the question; no query was truncated.")
    return encoded


def query_instruction(model_name: str) -> str:
    """Query-only policy; deliberately excluded from the document index fingerprint."""
    mode = os.getenv("QUERY_INSTRUCTION", "auto").strip().lower()
    if mode not in {"auto", "off"}:
        raise ValueError("QUERY_INSTRUCTION must be auto or off")
    return (BGE_QUERY_INSTRUCTION if mode == "auto"
            and model_name == "BAAI/bge-small-en-v1.5" else "")


def _enabled(name: str, default: str = "false") -> bool:
    from src.index_config import _bool
    try:
        return _bool(os.getenv(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} must be true or false") from exc


def _get_terminology(collection, snapshot, corpus, document_id=None):
    """Only explicit, initial-aligned parenthetical definitions are trusted.

    Cache all definitions by committed revision; scope BEFORE resolving ambiguity.
    No dictionary, language model, or uncommitted page text participates.
    """
    global _terminology_cache
    key = (getattr(collection, "name", id(collection)), snapshot.revision)
    if not _terminology_cache or _terminology_cache[0] != key:
        definitions = {}
        for record in corpus:
            text = record["text"] + "\n" + record["title"]
            # Restrict to alphabetic acronyms; identifiers and units are not aliases.
            for match in re.finditer(r"\b([A-Z][A-Z-]{1,9})\s*\(([^()\n]{3,100})\)|"
                                     r"([^()\n.!?;:]{3,100})\s*\(([A-Z][A-Z-]{1,9})\)", text):
                acronym = match[1] or match[4]
                raw = match[2] or match[3]
                letters = acronym.replace("-", "").lower()
                words = re.findall(r"[A-Za-z]+", raw)
                # Long-form-before-acronym may include sentence-leading prose.
                stopwords = {"of", "from", "and", "the", "for", "to", "in"}
                options = [words] if match[1] else [words[-n:] for n in range(len(letters), min(len(words), 2 * len(letters)) + 1)]
                words = next((candidate for candidate in options
                              if "".join(w[0].lower() for w in candidate) == letters
                              or "".join(w[0].lower() for w in candidate if w.lower() not in stopwords) == letters), [])
                if not words or re.search(r"\d", raw):
                    continue
                phrase = " ".join(words)
                entry = definitions.setdefault(acronym, {})
                entry.setdefault(phrase.casefold(), {"phrase": phrase, "papers": set()})["papers"].add(_paper_key(record))
        _terminology_cache = (key, definitions)
    resolved = {}
    for acronym, meanings in _terminology_cache[1].items():
        scoped = [value for value in meanings.values()
                  if document_id is None or (bool(document_id & value["papers"]) if isinstance(document_id, set) else document_id in value["papers"])]
        if len(scoped) == 1:
            resolved[acronym] = scoped[0]["phrase"]
    return resolved


def _search_queries(question, corrected, terminology, *, subqueries=()):
    """One shared variant allowance: plan queries, correction, then aliases.

    Subqueries are for trusted retrieval plans, never generated answers. The
    current pipeline has structural overview selection rather than subqueries.
    """
    limit = min(_non_negative_int("QUERY_MAX_VARIANTS", 2), 2)
    queries = [question]
    seen = {" ".join(question.split()).casefold()}

    def add(value):
        key = " ".join(value.split()).casefold()
        if value.strip() and key not in seen and len(queries) <= limit:
            queries.append(value)
            seen.add(key)

    for value in subqueries:
        add(value)
    add(corrected)
    if _enabled("QUERY_EXPANSION"):
        for acronym, phrase in sorted(terminology.items()):
            match = re.search(r"(?<![\w./+#-])" + re.escape(acronym) + r"(?![\w/+#-]|\.\w)", question)
            alias = phrase
            if not match:
                pattern = r"(?<![\w./+#-])" + r"[\s-]+".join(map(re.escape, phrase.split())) + r"(?![\w/+#-]|\.\w)"
                match = re.search(pattern, question, re.IGNORECASE)
                alias = acronym
            if match:
                # Insert only the attested alias; preserve every original character.
                add(question[:match.end()] + f" ({alias})" + question[match.end():])
    return queries


def _query_budgets(total, query_count):
    """Keep half for the original; all variants share the other half."""
    if query_count == 1:
        return [total]
    original = (total + 1) // 2
    quotient, remainder = divmod(total - original, query_count - 1)
    return [original] + [quotient + (i < remainder) for i in range(query_count - 1)]


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
        transform = os.getenv("RERANKER_SCORE_TRANSFORM", "model-default")
        if transform not in {"model-default", "raw", "sigmoid"}:
            raise ValueError("RERANKER_SCORE_TRANSFORM must be model-default, raw or sigmoid")
        if (reranker_model != "BAAI/bge-reranker-base" or transform != "model-default") and (
                "RERANKER_MIN_SCORE" not in os.environ and not os.getenv("RELEVANCE_CALIBRATION", "").strip()):
            raise ValueError("Incompatible implicit baseline threshold: set a model-specific "
                             "RERANKER_MIN_SCORE or RELEVANCE_CALIBRATION for the new model/activation")
        return cls(
            mode=mode,
            dense_candidates=_positive_int("DENSE_CANDIDATES", 20),
            bm25_candidates=_positive_int("BM25_CANDIDATES", 20),
            rrf_k=_positive_int("RRF_K", 60),
            rerank_candidates=_positive_int("RERANK_CANDIDATES", 20),
            final_results=_positive_int("FINAL_RESULTS", 5),
            reranker_model=reranker_model,
            reranker_min_score=_finite_float("RERANKER_MIN_SCORE", 0.01),
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


def _normalize_english_term(token: str) -> str:
    """Conservatively normalize common English inflections for lexical search."""
    if not token.isalpha() or len(token) < 4:
        return token

    if token.endswith("ies") and len(token) > 4:
        token = f"{token[:-3]}y"
    elif token.endswith("sses"):
        token = token[:-2]
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]

    # These forms drop the final "e" from verbs ending in "ute", so they
    # normalize contribute, contributes, contributing, and contributions alike.
    if token.endswith("ution"):
        return f"{token[:-5]}ute"
    if token.endswith("uting"):
        return f"{token[:-5]}ute"
    if token.endswith("uted"):
        return f"{token[:-4]}ute"
    return token


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"(?u)[^\W_][\w.+#/-]*", text.casefold())
    expanded = []
    for token in tokens:
        expanded.append(_normalize_english_term(token))
        expanded.extend(
            _normalize_english_term(part)
            for part in re.split(r"[.+#/-]+", token)
            if part != token
        )
    return expanded


def _correct_search_question(question: str, tokenized: list[list[str]]) -> str:
    """Correct confident corpus-backed typos for search, not for the answer."""
    counts = Counter(token for document in tokenized for token in document if token.isalpha())
    # Only ordinary retrieval vocabulary is eligible as a correction target.
    # Unknown names/technical terms are never guessed from arbitrary corpus words.
    ordinary = {"dimension", "trustworthiness", "retrieval", "generation", "evaluation",
                "hallucination", "contribute", "comparison", "performance", "methodology"}
    frequent = sorted(token for token, count in counts.items() if count >= 5 and token in ordinary)
    if not frequent:
        return question

    def correct(match: re.Match[str]) -> str:
        word = match.group(0)
        # Keep names and acronyms intact; sentence-initial capitalization is
        # allowed because it is common in ordinary questions.
        if not word.islower() or not word.isalpha() or len(word) < 6:
            return word
        normalized = _normalize_english_term(word.casefold())
        if counts[normalized] >= 1:
            return word
        matches = get_close_matches(normalized, frequent, n=2, cutoff=0.78)
        if not matches:
            return word
        best = matches[0]
        if len(matches) > 1:
            best_similarity = SequenceMatcher(None, normalized, best).ratio()
            next_similarity = SequenceMatcher(None, normalized, matches[1]).ratio()
            if best_similarity - next_similarity < 0.035:
                return word
        return best.capitalize() if word[0].isupper() else best

    return re.sub(r"(?u)(?<![\w./+#-])[^\W_][\w./+#-]*(?![\w./+#-])", correct, question)


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
        if (owner is None or (metadata.get("document_id") and metadata["document_id"] != owner["document_id"])
                or (owner.get("document_version") and metadata.get("document_version") != owner["document_version"])):
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


def _in_scope(key, scope):
    if scope is None:
        return True
    return key in scope if isinstance(scope, (set, frozenset)) else key == scope


def _dense_search(question: str, collection, count: int, candidate_count: int,
                  snapshot, document_id: str | None = None) -> list[dict]:
    if not snapshot.chunk_ids or candidate_count <= 0:
        return []
    config = get_index_config()
    model = get_embedding_model()
    encoded_question = _dense_query_text(question, model, config)
    query_embedding = model.encode(
        [encoded_question], normalize_embeddings=config.normalize_embeddings, prompt=""
    ).tolist()
    if len(query_embedding[0]) != config.embedding_dimension:
        raise ValueError(
            f"Query embedding dimension {len(query_embedding[0])} does not match "
            f"index dimension {config.embedding_dimension}; rebuild or correct configuration."
        )
    allowed_ids = (
        {chunk_id for chunk_id, owner in snapshot.owners.items()
         if _in_scope(owner["document_id"], document_id)}
        if document_id is not None else snapshot.chunk_ids
    )
    target = min(candidate_count, len(allowed_ids))
    fetch = min(max(target, 1), count)
    hits = []
    while fetch:
        query_args = dict(
            query_embeddings=query_embedding,
            n_results=fetch,
            include=["documents", "metadatas", "distances"],
        )
        if document_id is not None:
            query_args["where"] = {"document_id": {"$in": sorted(document_id)}} if isinstance(document_id, set) else {"document_id": document_id}
        results = collection.query(**query_args)
        hits = []
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0],
        ):
            owner = snapshot.owners.get(chunk_id)
            metadata = metadata or {}
            if (chunk_id not in allowed_ids or owner is None
                    or (metadata.get("document_id") and metadata["document_id"] != owner["document_id"])
                    or (owner.get("document_version") and metadata.get("document_version") != owner["document_version"])):
                continue
            hit = _make_committed_result(chunk_id, text, metadata, owner)
            hit["distance"] = hit["dense_distance"] = float(distance)
            hit["dense_score"] = -float(distance)
            hits.append(hit)
        if len(hits) >= target or fetch >= count:
            break
        fetch = min(count, max(fetch + 1, fetch * 2))
    return sorted(hits, key=lambda hit: (hit["distance"], hit["id"]))[:target]


def _bm25_search(question: str, collection, candidate_count: int, snapshot,
                 document_id: str | None = None) -> list[dict]:
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
        and _in_scope(_paper_key(records[i]), document_id)
    ]
    ranked = sorted(matching, key=lambda i: (-float(scores[i]), records[i]["id"]))
    hits = []
    for i in ranked[:candidate_count]:
        hit = records[i].copy()
        hit["bm25_score"] = float(scores[i])
        hits.append(hit)
    return hits


def reciprocal_rank_fusion(*ranked_lists: list[dict], k: int = 60,
                           weights: list[float] | None = None) -> list[dict]:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    weights = [1.0] * len(ranked_lists) if weights is None else weights
    if len(weights) != len(ranked_lists) or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Fusion weights must be finite, non-negative, and match ranked lists")
    fused = {}
    scores = {}
    for ranked, weight in zip(ranked_lists, weights):
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
                provenance = fused[chunk_id].get("retrieval_queries", []) + result.get("retrieval_queries", [])
                fused[chunk_id].update({
                    key: value for key, value in result.items()
                    if value is not None and key not in {"retrieval_queries", "dense_distance", "distance", "dense_score", "bm25_score"}
                })
                fused[chunk_id]["retrieval_queries"] = provenance
                for field in ("dense_distance", "distance", "dense_score", "bm25_score"):
                    values = [v for v in (fused[chunk_id].get(field), result.get(field)) if v is not None]
                    if values:
                        fused[chunk_id][field] = (min if field.endswith("distance") else max)(values)
            scores[chunk_id] += weight / (k + rank)

    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    for chunk_id in ordered_ids:
        fused[chunk_id]["rrf_score"] = scores[chunk_id]
    return [fused[chunk_id] for chunk_id in ordered_ids]


def _create_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    kwargs = {"device": "cpu", "max_length": 512}
    revision = os.getenv("RERANKER_REVISION", "").strip()
    if revision:
        kwargs["revision"] = revision
    return CrossEncoder(model_name, **kwargs)


def _get_reranker(model_name: str):
    global _reranker, _reranker_name
    cache_key = (model_name, os.getenv("RERANKER_REVISION", ""))
    if _reranker is None or _reranker_name != cache_key:
        _reranker = _create_reranker(model_name)
        _reranker_name = cache_key
    return _reranker


def _rerank(question: str, candidates: list[dict], model_name: str) -> list[dict]:
    if not candidates:
        return []
    model = _get_reranker(model_name)
    pairs = [(question, candidate["text"]) for candidate in candidates]
    contract = score_contract(model, model_name)
    kwargs = {}
    if contract["transformation"] != "model-default":
        from torch import nn
        kwargs["activation_fn"] = nn.Identity() if contract["transformation"] == "raw" else nn.Sigmoid()
    scores = model.predict(pairs, batch_size=16, show_progress_bar=False, **kwargs)
    if len(scores) != len(candidates) or any(not math.isfinite(float(v)) for v in scores):
        raise ValueError("Reranker returned invalid scalar scores")
    threshold = effective_threshold(contract, RetrievalConfig.from_env().reranker_min_score)
    ranked = sorted(
        zip(candidates, scores),
        key=lambda pair: (-float(pair[1]), -pair[0]["rrf_score"], pair[0]["id"]),
    )
    results = []
    for candidate, score in ranked:
        result = candidate.copy()
        result["reranker_score"] = float(score)
        result["score_contract"] = {**contract, "effective_threshold": threshold}
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


_OVERVIEW_QUESTION = re.compile(
    r"\b(?:summari[sz]e|summary|overview|contribut\w*|dimensions?|key findings|main findings|"
    r"key takeaways|main points|what (?:is|are) (?:this|the) (?:paper|survey|article) about)\b",
    re.IGNORECASE,
)
_OVERVIEW_CLAIM = re.compile(
    r"\b(?:we (?:propose|present|introduce|provide|review|survey|identify|discuss|examine|"
    r"evaluate|conclude)|this (?:paper|article|survey|work) (?:proposes|presents|introduces|"
    r"provides|reviews|surveys|discusses|examines|evaluates)|our contributions?)\b",
    re.IGNORECASE,
)
_OVERVIEW_TITLE_STOPWORDS = {"a", "an", "and", "in", "of", "the", "to", "this", "paper"}
_DIMENSION_LIST_QUESTION = re.compile(
    r"\b(?:what|which|list|name|identify)\b.*\bdimensions?\b", re.IGNORECASE
)
_DIRECT_DIMENSION_LIST = re.compile(
    r"\bdimensions?\b[^:.]{0,120}:\s*[^.]{15,}", re.IGNORECASE
)


def _overview_evidence(question: str, corpus: list[dict],
                       document_id: str | None = None) -> list[dict]:
    """Find short paper-level evidence whose relevance is structural, not lexical."""
    if not _OVERVIEW_QUESTION.search(question):
        return []

    by_paper: dict[str, list[dict]] = {}
    for result in corpus:
        key = _paper_key(result)
        if _in_scope(key, document_id):
            by_paper.setdefault(key, []).append(result)

    if document_id is None and len(by_paper) > 1:
        query_terms = set(_tokenize(question)) - _OVERVIEW_TITLE_STOPWORDS
        title_scores = {
            key: len(query_terms & (set(_tokenize(records[0]["title"])) - _OVERVIEW_TITLE_STOPWORDS))
            for key, records in by_paper.items()
        }
        best = max(title_scores.values())
        if best:
            by_paper = {key: records for key, records in by_paper.items()
                        if title_scores[key] == best}

    selected = []
    for records in by_paper.values():
        records = sorted(records, key=lambda item: (
            int(item["page"]), _chunk_position(item) or (int(item["page"]), 0), item["id"]
        ))
        if _DIMENSION_LIST_QUESTION.search(question):
            direct = next(
                (item for item in records
                 if _DIRECT_DIMENSION_LIST.search(item["text"])), None
            )
            if direct is not None:
                selected.append({**direct, "selection_reason": "direct_list", "adjacent_to": None})
                continue
        first_page = min(int(item["page"]) for item in records)
        opening = [item for item in records
                   if int(item["page"]) == first_page
                   and len(item["text"]) >= 120
                   and _OVERVIEW_CLAIM.search(item["text"])]
        opening.sort(key=lambda item: (
            -len(_OVERVIEW_CLAIM.findall(item["text"])),
            item["text"].count("@"),
            _chunk_position(item) or (int(item["page"]), 0),
        ))
        structural = opening[:1]
        for heading in ("abstract", "contribution", "conclusion", "introduction"):
            matches = [item for item in records
                       if heading in re.sub(r"[*_`#]", "", item.get("section", "")).casefold()
                       and len(item["text"]) >= 120]
            if matches:
                structural.append(matches[0])
        structural.extend(opening[1:2])
        seen = set()
        for item in structural:
            if item["id"] not in seen:
                selected.append({**item, "selection_reason": "overview", "adjacent_to": None})
                seen.add(item["id"])
    return selected


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
    """Threshold, diversify, and cap search anchors before context expansion."""
    relevant = [
        candidate for candidate in reranked
        if float(candidate["reranker_score"]) >= min_score
    ]
    if not relevant:
        return []

    anchors = _mmr_select(relevant, top_k, diversity, max_per_paper)
    if not anchors:
        return []
    # Neighboring context is added after ALL anchors have been selected.
    return [{**anchor, "selection_reason": "anchor", "adjacent_to": None}
            for anchor in anchors]


def _anchors_fit(question, anchors):
    """Conservative whole-anchor accounting before optional context is expanded."""
    from src.context_budget import generation_token_counter, positive_setting
    from src.generate import SYSTEM_PROMPT, build_context, build_user_prompt
    counter = generation_token_counter()
    return (counter.count(build_context(anchors)) <= positive_setting("CONTEXT_BUDGET_TOKENS", 4000)
            and counter.prompt_tokens(SYSTEM_PROMPT, build_user_prompt(question, anchors),
                positive_setting("CONTEXT_FRAMING_RESERVE", 32))
                + positive_setting("CONTEXT_ANSWER_RESERVE", 1024)
                + _non_negative_int("CONTEXT_RETRY_RESERVE", 256)
                <= positive_setting("CONTEXT_WINDOW_TOKENS", 8192))


def expand_context(question: str, anchors: list[dict], corpus: list[dict], snapshot,
                   *, tokenizer=None, radius: int | None = None,
                   budget: int | None = None, window: int | None = None,
                   answer_reserve: int | None = None,
                   expansion_budget: int | None = None,
                   same_section: bool | None = None) -> list[dict]:
    """Reserve all anchor intervals, then spend a separate budget on context.

    Evidence remains page-specific. Anchor scores and exact locations live in
    anchor_hits; related_anchor_ids also records shared, unranked neighbors.
    """
    from src.context_budget import TokenCounter, generation_token_counter
    from src.generate import build_context, build_user_prompt, SYSTEM_PROMPT

    counter = TokenCounter(tokenizer, "explicit-tokenizer", "") if tokenizer is not None else generation_token_counter()
    radius = _non_negative_int("CONTEXT_PARAGRAPHS", 1) if radius is None else radius
    budget = _positive_int("CONTEXT_BUDGET_TOKENS", 4000) if budget is None else budget
    window = _positive_int("CONTEXT_WINDOW_TOKENS", 8192) if window is None else window
    answer_reserve = _positive_int("CONTEXT_ANSWER_RESERVE", 1024) if answer_reserve is None else answer_reserve
    expansion_budget = (_non_negative_int("CONTEXT_EXPANSION_TOKENS", 1500)
                        if expansion_budget is None else expansion_budget)
    framing = _positive_int("CONTEXT_FRAMING_RESERVE", 32)
    retry_reserve = _non_negative_int("CONTEXT_RETRY_RESERVE", 256)
    if same_section is None:
        value = os.getenv("CONTEXT_SAME_SECTION", "true").lower()
        if value not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError("CONTEXT_SAME_SECTION must be true or false")
        same_section = value in {"true", "1", "yes"}
    values = (radius, budget, window, answer_reserve, expansion_budget)
    if any(not isinstance(v, int) or isinstance(v, bool) for v in values):
        raise ValueError("Context budgets and paragraph radius must be integers")
    if radius < 0 or expansion_budget < 0 or min(budget, window, answer_reserve) <= 0:
        raise ValueError("Context budgets must be positive; paragraph radius and expansion budget non-negative")
    fixed = counter.prompt_tokens(SYSTEM_PROMPT, build_user_prompt(question, []), framing)
    if fixed + retry_reserve + answer_reserve >= window:
        raise ValueError("CONTEXT_WINDOW_TOKENS leaves no space for evidence after system instructions, "
                         "question, CONTEXT_FRAMING_RESERVE, CONTEXT_RETRY_RESERVE and CONTEXT_ANSWER_RESERVE; "
                         "increase the window or reduce reserves/question length; no evidence was truncated.")
    if not anchors:
        return []
    # Stable deduplication before interval assembly; never re-select anchors here.
    unique_anchors = {}
    for item in anchors:
        unique_anchors.setdefault(item["id"], item)
    anchors = list(unique_anchors.values())

    def version_key(item):
        meta = item["metadata"]
        return (meta["document_id"], meta["document_version"])

    committed = {item["id"]: item for item in corpus
                 if item["id"] in snapshot.chunk_ids
                 and snapshot.owners.get(item["id"], {}).get("document_id") == item["metadata"].get("document_id")
                 and (not snapshot.owners[item["id"]].get("document_version")
                      or snapshot.owners[item["id"]]["document_version"] == item["metadata"].get("document_version"))}
    pages = {}
    for item in committed.values():
        meta = item["metadata"]
        if "page_context" not in meta or not meta.get("document_version"):
            continue
        owner = snapshot.owners[item["id"]]
        if meta.get("document_id") != owner["document_id"]:
            continue
        key = (*version_key(item), int(item["page"]))
        context = json.loads(meta["page_context"])
        if key in pages and pages[key] != context:
            raise ValueError("Committed page context is inconsistent; rebuild the index")
        pages[key] = context
    intervals = []
    for rank, anchor in enumerate(anchors):
        meta = anchor["metadata"]
        if anchor["id"] not in committed:
            raise ValueError("Cannot expand an uncommitted search hit")
        if not meta.get("document_version") or "page_context" not in meta:
            raise ValueError("Search hit has no versioned paragraph context; rebuild the index")
        stored = committed[anchor["id"]]
        if stored["text"] != anchor["text"] or stored["metadata"] != meta:
            raise ValueError("Search hit and committed record disagree; retry retrieval")
        key = (*version_key(anchor), int(anchor["page"]))
        if key not in pages:
            raise ValueError("Search hit has no committed paragraph context; rebuild the index")
        start, end = meta["character_start"], meta["character_end"]
        if pages[key]["text"][start:end] != anchor["text"]:
            raise ValueError("Search hit and stored context disagree; rebuild the index")
        intervals.append((key, start, end, rank))

    def render(spans):
        groups = []

        def section_at(key, offset):
            return next((para["section_id"] for para in pages[key]["paragraphs"]
                         if para["start"] <= offset < para["end"]), None)

        for key in dict.fromkeys(span[0] for span in spans):
            ordered = sorted((span for span in spans if span[0] == key), key=lambda x: (x[1], x[2]))
            for _, start, end, rank in ordered:
                if (groups and groups[-1][0] == key
                        and (start <= groups[-1][2]
                             or (not pages[key]["text"][groups[-1][2]:start].strip()
                                 and section_at(key, start) == section_at(key, groups[-1][2] - 1)))):
                    old_key, old_start, old_end, ranks = groups[-1]
                    groups[-1] = (key, old_start, max(end, old_end), ranks | {rank})
                else:
                    groups.append((key, start, end, {rank}))
        groups.sort(key=lambda x: (min(x[3]), x[0][2], x[1]))
        result = []
        for key, start, end, ranks in groups:
            anchor = anchors[min(ranks)]
            contained = [r for r, (anchor_key, left, right, _) in enumerate(anchor_intervals)
                         if anchor_key == key and start <= left and right <= end]
            text = pages[key]["text"][start:end]
            meta = {**anchor["metadata"], "page": key[2],
                    "character_start": start, "character_end": end}
            # Offsets and paragraph locations always describe the expanded text.
            covered = [i for i, para in enumerate(pages[key]["paragraphs"])
                       if para["start"] < end and para["end"] > start]
            if covered:
                meta.update(paragraph_start=min(covered), paragraph_end=max(covered),
                            section=pages[key]["paragraphs"][covered[0]]["section"],
                            section_id=pages[key]["paragraphs"][covered[0]]["section_id"])
            meta.pop("page_context", None)
            meta.pop("chunk_index", None)
            evidence = {**anchor, "id": f"{key[0]}-{key[1]}-p{key[2]}-{start}:{end}",
                        "text": text, "page": key[2], "section": meta["section"],
                        "metadata": meta, "anchor_id": anchor["id"],
                        "anchor_hits": [{"id": anchors[r]["id"], "page": anchors[r]["page"],
                                         "text": anchors[r]["text"],
                                         "character_start": anchors[r]["metadata"]["character_start"],
                                         "character_end": anchors[r]["metadata"]["character_end"],
                                         "rank": r + 1,
                                         "selection_reason": anchors[r].get("selection_reason", "anchor"),
                                         "retrieval_queries": anchors[r].get("retrieval_queries", []),
                                         "scores": {k: v for k, v in anchors[r].items()
                                                    if k.endswith(("score", "distance"))}}
                                        for r in sorted(ranks)]}
            evidence["anchor_hits"] = [hit for hit in evidence["anchor_hits"]
                                       if hit["id"] in {anchors[r]["id"] for r in contained}]
            evidence["retrieval_queries"] = [query for hit in evidence["anchor_hits"]
                                             for query in hit["retrieval_queries"]]
            # A context block is not an independently ranked search result.
            exact = len(contained) == 1 and text == anchors[contained[0]]["text"]
            for field in list(evidence):
                if field.endswith(("score", "distance")):
                    evidence[field] = anchors[contained[0]].get(field) if exact else None
            evidence.pop("anchor_rank", None)
            evidence["group_rank"] = min(ranks) + 1
            evidence["is_anchor"] = bool(contained)
            evidence["evidence_kind"] = "anchor" if exact else ("anchor_context" if contained else "expansion")
            evidence["related_anchor_ids"] = [anchors[r]["id"] for r in sorted(ranks | set(contained))]
            evidence["anchor_id"] = anchors[contained[0]]["id"] if contained else None
            evidence["adjacent_to"] = evidence["related_anchor_ids"] if not contained else None
            evidence["selection_reason"] = evidence["evidence_kind"]
            meta["evidence_kind"] = evidence["evidence_kind"]
            meta["related_anchor_ids"] = evidence["related_anchor_ids"]
            meta["anchor_hits"] = evidence["anchor_hits"]
            result.append(evidence)
        return result

    anchor_intervals = list(intervals)

    def sizes(spans):
        result = render(spans)
        evidence_tokens = counter.count(build_context(result))
        prompt_tokens = counter.prompt_tokens(SYSTEM_PROMPT, build_user_prompt(question, result), framing) + retry_reserve
        return evidence_tokens, prompt_tokens

    anchor_tokens, anchor_prompt_tokens = sizes(intervals)
    if anchor_tokens > budget or anchor_prompt_tokens + answer_reserve > window:
        raise ValueError(f"Selected anchors exceed the context budget: evidence={anchor_tokens}/{budget}, "
                         f"prompt+answer={anchor_prompt_tokens + answer_reserve}/{window} tokens "
                         f"({counter.method}). Increase CONTEXT_BUDGET_TOKENS/CONTEXT_WINDOW_TOKENS, "
                         "shorten the question, or reduce top_k; no evidence was truncated.")

    def fits(spans):
        evidence_tokens, prompt_tokens = sizes(spans)
        return (evidence_tokens <= budget and evidence_tokens - anchor_tokens <= expansion_budget
                and prompt_tokens + answer_reserve <= window)

    # Stronger anchor groups spend the optional budget first, nearest context first.
    choices = []
    for rank, anchor in enumerate(anchors):
        meta = anchor["metadata"]
        version = version_key(anchor)
        ordered = [(key, pid, para) for key, context in sorted(pages.items())
                   if key[:2] == version for pid, para in enumerate(context["paragraphs"])]
        positions = [i for i, (key, pid, para) in enumerate(ordered)
                     if key[2] == anchor["page"]
                     and meta["paragraph_start"] <= pid <= meta["paragraph_end"]]
        if not positions:
            continue
        for i, (key, pid, para) in enumerate(ordered):
            distance = min(abs(i - pos) for pos in positions)
            if radius > 0 and expansion_budget > 0 and distance <= radius and (not same_section or para["section_id"] == meta["section_id"]):
                choices.append((rank, distance, key, para["start"], para["end"]))
    for rank, distance, key, start, end in sorted(choices):
        candidate = intervals + [(key, start, end, rank)]
        if fits(candidate):
            intervals = candidate
    result = render(intervals)
    evidence_tokens, prompt_tokens = sizes(intervals)
    usage = {"token_count_method": counter.method, "token_count_limitation": counter.limitation,
             "anchor_tokens": anchor_tokens, "evidence_tokens": evidence_tokens,
             "expansion_tokens": max(0, evidence_tokens - anchor_tokens),
             "prompt_tokens": prompt_tokens, "answer_reserve": answer_reserve,
             "window_tokens": window, "anchor_count": len(anchors),
             "evidence_budget": budget, "expansion_budget": expansion_budget,
             "framing_reserve": framing, "retry_reserve": retry_reserve}
    for item in result:
        item["metadata"]["context_usage"] = usage
    return result


_THIS_PAPER = re.compile(r"\bthis\s+(?:paper|article|study|survey)\b", re.IGNORECASE)
_THIS_PAPER_MESSAGES = (
    "Which paper, boss? frog cannot read minds yet.",
    "Frog requires target paper. otherwise frog just vibes.",
    "Choose a paper first. the hat gives knowledge, not telepathy."
)
SELECTED_PAPER_NOT_INDEXED_MESSAGE = (
    "Frog sees the paper, but has not read it yet. sync the library first."
)


def _retrieve_locked(question: str, top_k: int | None = None,
                     retrieval_mode: str | None = None,
                     paper: str | None = None, *, debug: dict | None = None,
                     plan: RetrievalPlan | None = None) -> list[dict]:
    """Return ranked Chroma chunks with citation metadata and method scores."""
    config = RetrievalConfig.from_env()
    plan = plan or infer_plan(question, (paper,) if paper else ())
    if paper and plan.papers and tuple(plan.papers) != (paper,):
        raise ValueError("paper and plan scope disagree")
    policy = os.getenv("PAPER_CAP_POLICY", "baseline")
    if policy not in {"baseline", "task-aware"}:
        raise ValueError("PAPER_CAP_POLICY must be baseline or task-aware")
    if policy == "task-aware" and (retrieval_mode or config.mode) != "hybrid-rerank":
        raise ValueError("Task-aware policy requires relevance reranking")
    instruction = query_instruction(get_index_config().embedding_model)
    if debug is not None:
        debug.clear()
        debug.update(original_query=question, variants=[], query_instruction=instruction,
                     instruction_used=False, expansion_enabled=_enabled("QUERY_EXPANSION"),
                     candidate_counts=[])
    top_k = config.final_results if top_k is None else top_k
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    mode = retrieval_mode or config.mode
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"retrieval_mode must be one of: {', '.join(RETRIEVAL_MODES)}")
    if paper is None and not plan.papers and _THIS_PAPER.search(question):
        raise ValueError(random.choice(_THIS_PAPER_MESSAGES))

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    snapshot = _get_committed_snapshot(collection)
    if not snapshot.chunk_ids:
        return []

    document_id = None
    if paper is not None or plan.papers:
        scoped = set()
        for path in ((paper,) if paper else plan.papers):
            ids = {owner["document_id"] for owner in snapshot.owners.values()
                   if path in owner.get("paths", [])}
            if len(ids) != 1:
                raise ValueError(SELECTED_PAPER_NOT_INDEXED_MESSAGE)
            scoped.update(ids)
        document_id = next(iter(scoped)) if len(scoped) == 1 else scoped

    _, correction_corpus, tokenized = _get_bm25_index(collection, snapshot)
    if document_id is not None:
        tokenized = [tokens for result, tokens in zip(correction_corpus, tokenized)
                     if _in_scope(_paper_key(result), document_id)]
    corrected = _correct_search_question(question, tokenized)
    terminology = (_get_terminology(collection, snapshot, correction_corpus, document_id)
                   if _enabled("QUERY_EXPANSION") else {})
    queries = (_search_queries(question, corrected, terminology, subqueries=plan.subqueries)
               if plan.subqueries else _search_queries(question, corrected, terminology))
    if len(queries) > 1:
        approved = []
        model, index_config = get_embedding_model(), get_index_config()
        for i, query in enumerate(queries):
            try:
                _dense_query_text(query, model, index_config)
            except QueryTokenLimitError as exc:
                if i == 0:
                    raise
                if debug is not None:
                    debug.setdefault("rejected_variants", []).append({"query": query, "reason": str(exc)})
            else:
                approved.append(query)
        queries = approved
    dense_budgets = _query_budgets(config.dense_candidates, len(queries))
    bm25_budgets = _query_budgets(config.bm25_candidates, len(queries))
    rankings, weights = [], []
    for i, search_question in enumerate(queries):
        dense = _dense_search(search_question, collection, count, dense_budgets[i],
                              snapshot, document_id)
        if debug is not None and dense_budgets[i] > 0:
            debug["instruction_used"] = bool(instruction)
        bm25 = ([] if mode == "dense" or bm25_budgets[i] == 0 else
                _bm25_search(search_question, collection, bm25_budgets[i], snapshot, document_id))
        for method, hits in (("dense", dense), ("bm25", bm25)):
            for rank, hit in enumerate(hits, 1):
                hit["retrieval_queries"] = [{"query": search_question, "original": i == 0,
                                             "method": method, "rank": rank,
                                             "score": hit.get(method + "_score")}]
            if mode != "dense" or method == "dense":
                rankings.append(hits)
                weights.append(1.0 if i == 0 else 1.0 / (len(queries) - 1))
        if debug is not None:
            debug["candidate_counts"].append({"query": search_question, "dense": len(dense),
                                               "bm25": len(bm25), "dense_budget": dense_budgets[i],
                                               "bm25_budget": 0 if mode == "dense" else bm25_budgets[i]})
    fused = reciprocal_rank_fusion(*rankings, k=config.rrf_k, weights=weights)
    if debug is not None:
        debug.update(query_embedding_calls=sum(b > 0 for b in dense_budgets), variants=queries[1:], committed_revision=snapshot.revision,
                     fused_candidates=len(fused), reranked_candidates=0)
        debug["candidate_provenance"] = {
            hit["id"]: {"queries": hit.get("retrieval_queries", []),
                        "rrf_score": hit["rrf_score"], "page": hit["page"],
                        "source": hit["source"]} for hit in fused}
    if mode == "dense":
        results = dense if len(queries) == 1 else fused
    else:
        if mode == "hybrid":
            results = fused
        else:
            candidates = fused[:config.rerank_candidates]
            # Reserve half the rerank allowance for original-query hits, without
            # adding reranker calls or allowing aliases to crowd them all out.
            original_hits = [hit for hit in fused if any(q["original"] for q in hit.get("retrieval_queries", []))]
            reserved = original_hits[:(config.rerank_candidates + 1) // 2]
            candidates = list({hit["id"]: hit for hit in reserved + candidates}.values())[:config.rerank_candidates]
            if policy == "task-aware":
                _, structural_corpus, _ = _get_bm25_index(collection, snapshot)
                structural_corpus = [c for c in structural_corpus if _in_scope(_paper_key(c), document_id)]
                structural = _overview_evidence(question, structural_corpus, document_id)
                # Keep the original-query reservation and share the remaining allowance
                # with structural overview candidates.
                ordered = reserved + structural + candidates
                candidates = list({c["id"]: c for c in ordered}.values())[:config.rerank_candidates]
                structural_ids = {c["id"] for c in structural}
                for c in candidates:
                    if c.get("rrf_score") is None:
                        c["rrf_score"] = 0.0
                    if c["id"] in structural_ids and not c.get("retrieval_queries"):
                        c["retrieval_queries"] = [{"query": question, "original": True,
                            "method": "structural", "rank": None, "score": None}]
            reranked = _rerank(question, candidates, config.reranker_model)
            if debug is not None:
                debug["reranked_candidates"] = len(candidates)
                debug["reranker_calls"] = int(bool(candidates))
                debug["reranked_scores"] = {hit["id"]: hit["reranker_score"] for hit in reranked}
            _, corpus, _ = _get_bm25_index(collection, snapshot)
            if document_id is not None:
                corpus = [result for result in corpus if _in_scope(_paper_key(result), document_id)]
            max_per_paper = config.max_chunks_per_paper
            results = _select_context(
                reranked,
                corpus,
                top_k=top_k,
                min_score=reranked[0].get("score_contract", {}).get("effective_threshold", config.reranker_min_score) if reranked else config.reranker_min_score,
                max_per_paper=max_per_paper,
                diversity=config.mmr_lambda,
                adjacent_chunks=config.adjacent_chunks,
            )
            overview = _overview_evidence(question, corpus, document_id)
            for item in overview:
                item["retrieval_queries"] = [{"query": question, "original": True,
                                              "method": "structural", "rank": None, "score": None}]
            if overview and policy == "baseline" and not os.getenv("RELEVANCE_CALIBRATION", "").strip():
                if _DIMENSION_LIST_QUESTION.search(question) and all(
                    item["selection_reason"] == "direct_list" for item in overview
                ):
                    results = overview[:top_k]
                else:
                    target_papers = {_paper_key(item) for item in overview}
                    selected_ids = set()
                    paper_counts: dict[str, int] = {}
                    focused = []
                    for item in overview + results:
                        key = _paper_key(item)
                        if (item["id"] in selected_ids or key not in target_papers
                                or paper_counts.get(key, 0) >= max_per_paper
                                or len(focused) >= top_k):
                            continue
                        focused.append(item)
                        selected_ids.add(item["id"])
                        paper_counts[key] = paper_counts.get(key, 0) + 1
                    results = focused

    if mode == "hybrid-rerank":
        threshold = reranked[0].get("score_contract", {}).get("effective_threshold", config.reranker_min_score) if reranked else config.reranker_min_score
        if policy == "task-aware":
            # Existing structural overview supplies candidates, never bypasses relevance.
            relevant = sorted([c for c in reranked if c["reranker_score"] >= threshold],
                              key=lambda c: (-c["reranker_score"], c["id"]))
            if plan.papers:
                requested_sources = {owner["document_id"]: path for path in plan.papers
                                     for owner in snapshot.owners.values() if path in owner.get("paths", [])}
                relevant = [{**c, "source": requested_sources.get(_paper_key(c), c["source"])}
                            for c in relevant]
            results = select_anchors(relevant, plan, top_k, config.max_chunks_per_paper,
                                     _paper_key, debug, fits=lambda chosen: _anchors_fit(question, chosen))
        if debug is not None:
            debug["score_contract"] = reranked[0].get("score_contract", {}) if reranked else {}
            debug["effective_threshold"] = threshold
            debug["rejected_candidates"] = [{"id": c["id"], "score": c["reranker_score"],
                                              "reason": "below_threshold"}
                                             for c in reranked if c["reranker_score"] < threshold]
    counts = Counter()
    anchors = []
    for result in results:
        key = _paper_key(result)
        if (policy == "task-aware" or counts[key] < config.max_chunks_per_paper) and len(anchors) < top_k:
            anchors.append({**result, "is_anchor": True, "anchor_rank": len(anchors) + 1})
            counts[key] += 1
    if debug is not None:
        debug["selected_anchors"] = [{"id": hit["id"], "source": hit["source"],
                                      "page": hit["page"], "rank": hit["anchor_rank"],
                                      "retrieval_queries": hit.get("retrieval_queries", [])}
                                     for hit in anchors]
    results = expand_context(question, anchors, correction_corpus, snapshot)

    for result in results:
        result["metadata"]["retrieval_scope"] = {"plan": asdict(plan), "mode": mode,
            "anchor_budget": top_k, "committed_revision": snapshot.revision}
    if paper is not None:
        return [
            {**result, "source": paper,
             "metadata": {**result["metadata"], "source": paper}}
            for result in results
        ]
    return results


def retrieve(question: str, top_k: int | None = None,
             retrieval_mode: str | None = None,
             paper: str | None = None, *, debug: dict | None = None,
                     plan: RetrievalPlan | None = None) -> list[dict]:
    """Search a consistent committed manifest snapshot in every retrieval mode."""
    from src.sync import INDEX_LOCK

    with INDEX_LOCK:
        return _retrieve_locked(question, top_k=top_k,
                                retrieval_mode=retrieval_mode, paper=paper, debug=debug, plan=plan)
