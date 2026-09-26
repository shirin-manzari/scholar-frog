import sys
from types import SimpleNamespace

import pytest

import src.retrieve as retrieval
from src.generate import build_context


class FakeCollection:
    def __init__(self, rows):
        self.rows = {row["id"]: row for row in rows}
        self.dense_order = list(self.rows)

    def count(self):
        return len(self.rows)

    def get(self, include):
        rows = [self.rows[chunk_id] for chunk_id in self.rows]
        return {
            "ids": [row["id"] for row in rows],
            "documents": [row["text"] for row in rows],
            "metadatas": [row["metadata"] for row in rows],
        }

    def query(self, query_embeddings, n_results, include):
        rows = [self.rows[chunk_id] for chunk_id in self.dense_order[:n_results]]
        return {
            "ids": [[row["id"] for row in rows]],
            "documents": [[row["text"] for row in rows]],
            "metadatas": [[row["metadata"] for row in rows]],
            "distances": [[row["distance"] for row in rows]],
        }


@pytest.fixture(autouse=True)
def reset_retrieval_state(monkeypatch):
    for name in (
        "RETRIEVAL_MODE", "DENSE_CANDIDATES", "BM25_CANDIDATES", "RRF_K",
        "RERANK_CANDIDATES", "FINAL_RESULTS", "RERANKER_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    retrieval._bm25_cache = None
    retrieval._reranker = None
    retrieval._reranker_name = None


@pytest.fixture
def collection():
    rows = [
        {
            "id": "chunk-a",
            "text": "BGE-reranker-base scores a RAG retrieval passage.",
            "metadata": {
                "source": "paper-a.pdf", "title": "Paper A", "section": "Methods",
                "page": 2, "file_hash": "hash-a",
            },
            "distance": 0.1,
        },
        {
            "id": "chunk-b",
            "text": "A different dense model retrieves academic passages.",
            "metadata": {
                "source": "paper-b.pdf", "title": "Paper B", "section": "Results",
                "page": 7, "file_hash": "hash-b",
            },
            "distance": 0.2,
        },
        {
            "id": "chunk-c",
            "text": "Retrieval augmented generation can produce hallucinations.",
            "metadata": {
                "source": "paper-c.pdf", "title": "Paper C", "section": "Limitations",
                "page": 4, "file_hash": "hash-c",
            },
            "distance": 0.3,
        },
    ]
    return FakeCollection(rows)


class FakeEmbeddingModel:
    def encode(self, questions, **kwargs):
        class Embeddings(list):
            def tolist(self):
                return list(self)

        return Embeddings([[0.1, 0.2] for _ in questions])


def test_bm25_returns_exact_academic_term_match(collection):
    hits = retrieval._bm25_search("BGE-reranker-base", collection, candidate_count=3)

    assert hits[0]["id"] == "chunk-a"
    assert hits[0]["bm25_score"] is not None
    assert all("BGE-reranker-base" in hit["text"] for hit in hits)


def test_bm25_matches_components_of_hyphenated_terms(collection):
    hits = retrieval._bm25_search("BGE reranker base", collection, candidate_count=3)

    assert hits[0]["id"] == "chunk-a"


def test_bm25_breaks_equal_score_ties_by_chunk_id():
    rows = [
        {"id": "z", "text": "shared-term appears here", "metadata": {
            "source": "z.pdf", "title": "Z", "page": 1,
        }},
        {"id": "a", "text": "shared-term appears there", "metadata": {
            "source": "a.pdf", "title": "A", "page": 1,
        }},
        {"id": "other", "text": "unrelated vocabulary only", "metadata": {
            "source": "other.pdf", "title": "Other", "page": 1,
        }},
    ]

    hits = retrieval._bm25_search("shared-term", FakeCollection(rows), 3)

    assert [hit["id"] for hit in hits[:2]] == ["a", "z"]


def test_bm25_index_is_reused_until_chroma_content_changes(collection):
    first = retrieval._get_bm25_index(collection)[0]
    assert retrieval._get_bm25_index(collection)[0] is first

    collection.rows["chunk-a"]["text"] = "Changed document text after reingestion."
    rebuilt = retrieval._get_bm25_index(collection)[0]

    assert rebuilt is not first


def test_rrf_combines_overlapping_and_non_overlapping_results():
    dense = [{"id": "a"}, {"id": "b"}]
    bm25 = [{"id": "a"}, {"id": "c"}]

    fused = retrieval.reciprocal_rank_fusion(dense, bm25, k=60)

    assert [item["id"] for item in fused] == ["a", "b", "c"]
    assert fused[0]["rrf_score"] == pytest.approx(2 / 61)
    assert fused[1]["rrf_score"] == pytest.approx(1 / 62)
    assert fused[2]["rrf_score"] == pytest.approx(1 / 62)


def test_rrf_deduplicates_chunk_ids_within_and_across_rankings():
    fused = retrieval.reciprocal_rank_fusion(
        [{"id": "same"}, {"id": "same"}], [{"id": "same"}]
    )

    assert [item["id"] for item in fused] == ["same"]
    assert fused[0]["rrf_score"] == pytest.approx(2 / 61)


def test_rrf_uses_chunk_id_for_deterministic_ties():
    fused = retrieval.reciprocal_rank_fusion([{"id": "z"}], [{"id": "a"}], k=60)

    assert [item["id"] for item in fused] == ["a", "z"]


def test_dense_hybrid_and_hybrid_rerank_modes_preserve_metadata(
    collection, monkeypatch
):
    monkeypatch.setattr(retrieval, "get_collection", lambda: collection)
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: FakeEmbeddingModel())
    created = []
    monkeypatch.setenv("RERANK_CANDIDATES", "2")
    rerank_batch_sizes = []

    class FakeCrossEncoder:
        def predict(self, pairs, batch_size, show_progress_bar):
            rerank_batch_sizes.append(len(pairs))
            return [0.1 if text.startswith("BGE") else 0.9 for _, text in pairs]

    monkeypatch.setattr(
        retrieval, "_create_reranker", lambda model_name: created.append(model_name) or FakeCrossEncoder()
    )

    dense = retrieval.retrieve("RAG", top_k=2, retrieval_mode="dense")
    hybrid = retrieval.retrieve("BGE-reranker-base", top_k=2, retrieval_mode="hybrid")
    reranked = retrieval.retrieve(
        "BGE-reranker-base", top_k=2, retrieval_mode="hybrid-rerank"
    )

    assert len(dense) == len(hybrid) == len(reranked) == 2
    assert dense[0]["id"] == "chunk-a"
    assert hybrid[0]["rrf_score"] is not None
    assert reranked[0]["reranker_score"] == 0.9
    assert reranked[0]["id"] == "chunk-b"
    assert reranked[0]["metadata"]["source"] == "paper-b.pdf"
    assert reranked[0]["page"] == 7
    assert created == ["BAAI/bge-reranker-base"]
    assert rerank_batch_sizes == [2]


def test_reranker_is_cached_and_limited_to_cpu_model_input(monkeypatch):
    created = []
    create_reranker = retrieval._create_reranker

    class FakeCrossEncoder:
        def predict(self, pairs, batch_size, show_progress_bar):
            return [0.5 for _ in pairs]

    monkeypatch.setattr(
        retrieval, "_create_reranker",
        lambda model_name: created.append(model_name) or FakeCrossEncoder(),
    )
    candidate = {
        "id": "x", "text": "Long text", "rrf_score": 0.02,
        "metadata": {"source": "x.pdf"},
    }

    retrieval._rerank("question", [candidate], "model-a")
    retrieval._rerank("question", [candidate], "model-a")

    assert created == ["model-a"]

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=lambda model_name, **kwargs: calls.append((model_name, kwargs))),
    )
    create_reranker("model-b")
    assert calls == [("model-b", {"device": "cpu", "max_length": 512})]


def test_empty_collection_and_unmatched_bm25_query_return_no_results(monkeypatch):
    empty = FakeCollection([])
    monkeypatch.setattr(retrieval, "get_collection", lambda: empty)
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: pytest.fail("loaded"))

    for mode in retrieval.RETRIEVAL_MODES:
        assert retrieval.retrieve("anything", retrieval_mode=mode) == []
    assert retrieval._bm25_search("anything", empty, 5) == []
    assert retrieval._bm25_search("... !!!", empty, 5) == []


@pytest.mark.parametrize("name,value", [("RRF_K", "0"), ("DENSE_CANDIDATES", "bad")])
def test_invalid_numeric_configuration_is_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        retrieval.RetrievalConfig.from_env()


def test_invalid_retrieval_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "lexical-only")

    with pytest.raises(ValueError, match="RETRIEVAL_MODE"):
        retrieval.RetrievalConfig.from_env()


def test_empty_reranker_model_is_rejected(monkeypatch):
    monkeypatch.setenv("RERANKER_MODEL", " ")

    with pytest.raises(ValueError, match="RERANKER_MODEL"):
        retrieval.RetrievalConfig.from_env()


def test_candidate_counts_and_final_results_are_configurable(monkeypatch):
    monkeypatch.setenv("DENSE_CANDIDATES", "12")
    monkeypatch.setenv("BM25_CANDIDATES", "13")
    monkeypatch.setenv("RERANK_CANDIDATES", "14")
    monkeypatch.setenv("FINAL_RESULTS", "4")

    config = retrieval.RetrievalConfig.from_env()

    assert (config.dense_candidates, config.bm25_candidates) == (12, 13)
    assert (config.rerank_candidates, config.final_results) == (14, 4)


def test_generation_context_uses_retrieved_citation_metadata(collection, monkeypatch):
    monkeypatch.setattr(retrieval, "get_collection", lambda: collection)
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: FakeEmbeddingModel())
    hit = retrieval.retrieve("RAG", top_k=1, retrieval_mode="dense")[0]

    context = build_context([hit])

    assert context.startswith("[Paper A, Methods, p.2]")
    assert hit["metadata"]["file_hash"] == "hash-a"
