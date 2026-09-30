import sys
from types import SimpleNamespace

import pytest

import src.retrieve as retrieval
from src.sync import CommittedSnapshot
from src.generate import build_context


class FakeCollection:
    def __init__(self, rows):
        self.rows = {row["id"]: row for row in rows}
        self.dense_order = list(self.rows)

    def count(self):
        return len(self.rows)

    def get(self, include, ids=None):
        rows = [self.rows[chunk_id] for chunk_id in (ids or self.rows) if chunk_id in self.rows]
        return {
            "ids": [row["id"] for row in rows],
            "documents": [row["text"] for row in rows],
            "metadatas": [row["metadata"] for row in rows],
        }

    def query(self, query_embeddings, n_results, include, where=None):
        rows = [self.rows[chunk_id] for chunk_id in self.dense_order]
        if where is not None:
            rows = [row for row in rows if row["metadata"].get("document_id", row["metadata"].get("file_hash")) == where["document_id"]]
        rows = rows[:n_results]
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
        "RERANKER_MIN_SCORE", "MAX_CHUNKS_PER_PAPER", "MMR_LAMBDA",
        "ADJACENT_CHUNKS",
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


def committed_snapshot(collection):
    owners = {}
    for chunk_id, row in collection.rows.items():
        metadata = row["metadata"]
        owner_id = metadata.get("document_id", metadata.get("file_hash"))
        owners[chunk_id] = {"document_id": owner_id, "paths": [metadata["source"]]}
    return CommittedSnapshot("snapshot-" + str(len(owners)), frozenset(owners), owners)


class FakeEmbeddingModel:
    def encode(self, questions, **kwargs):
        class Embeddings(list):
            def tolist(self):
                return list(self)

        return Embeddings([[0.1, 0.2] for _ in questions])


def test_bm25_returns_exact_academic_term_match(collection):
    hits = retrieval._bm25_search("BGE-reranker-base", collection, candidate_count=3,
                                  snapshot=committed_snapshot(collection))

    assert hits[0]["id"] == "chunk-a"
    assert hits[0]["bm25_score"] is not None
    assert all("BGE-reranker-base" in hit["text"] for hit in hits)


def test_bm25_matches_components_of_hyphenated_terms(collection):
    hits = retrieval._bm25_search("BGE reranker base", collection, candidate_count=3,
                                  snapshot=committed_snapshot(collection))

    assert hits[0]["id"] == "chunk-a"


@pytest.mark.parametrize(
    "term",
    ["contribute", "contributes", "contributed", "contributing", "contribution", "contributions"],
)
def test_tokenizer_normalizes_contribute_word_family(term):
    assert retrieval._tokenize(term) == ["contribute"]


def test_bm25_matches_morphological_variants():
    rows = [
        {"id": "contributions", "text": "The survey's contributions are a taxonomy and benchmarks.", "metadata": {
            "source": "survey.pdf", "title": "Survey", "page": 1,
        }},
        {"id": "other", "text": "This passage discusses unrelated limitations.", "metadata": {
            "source": "other.pdf", "title": "Other", "page": 1,
        }},
    ]
    collection = FakeCollection(rows)

    hits = retrieval._bm25_search(
        "What does the survey contribute?", collection, 2, committed_snapshot(collection)
    )

    assert [hit["id"] for hit in hits] == ["contributions"]


def test_search_corrects_confident_typos_from_paper_vocabulary():
    tokens = [["dimension"] * 6 + ["trustworthiness"] * 6]

    assert retrieval._correct_search_question(
        "What are diminention of trustworthiness?", tokens
    ) == "What are dimension of trustworthiness?"
    assert retrieval._correct_search_question(
        "What is quantum entanglement?", tokens
    ) == "What is quantum entanglement?"


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

    collection = FakeCollection(rows)
    hits = retrieval._bm25_search("shared-term", collection, 3,
                                  snapshot=committed_snapshot(collection))

    assert [hit["id"] for hit in hits[:2]] == ["a", "z"]


def test_bm25_index_is_reused_until_chroma_content_changes(collection):
    snapshot = committed_snapshot(collection)
    first = retrieval._get_bm25_index(collection, snapshot)[0]
    assert retrieval._get_bm25_index(collection, snapshot)[0] is first

    collection.rows["chunk-a"]["text"] = "Changed document text after reingestion."
    changed_snapshot = CommittedSnapshot("after-commit", snapshot.chunk_ids, snapshot.owners)
    rebuilt = retrieval._get_bm25_index(collection, changed_snapshot)[0]

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
    monkeypatch.setattr(retrieval, "get_index_config", lambda: __import__("src.index_config", fromlist=["IndexConfig"]).IndexConfig(embedding_dimension=2))
    monkeypatch.setattr(retrieval, "_get_committed_snapshot", committed_snapshot)
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


def test_selected_paper_scopes_every_retrieval_mode_and_uses_selected_alias(
    collection, monkeypatch
):
    monkeypatch.setattr(retrieval, "get_collection", lambda: collection)
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(
        retrieval, "get_index_config",
        lambda: __import__("src.index_config", fromlist=["IndexConfig"]).IndexConfig(
            embedding_dimension=2
        ),
    )

    def snapshot_with_alias(collection):
        snapshot = committed_snapshot(collection)
        snapshot.owners["chunk-c"]["paths"].append("copy-of-c.pdf")
        return snapshot

    monkeypatch.setattr(retrieval, "_get_committed_snapshot", snapshot_with_alias)

    class FakeCrossEncoder:
        def predict(self, pairs, batch_size, show_progress_bar):
            return [0.8 for _ in pairs]

    monkeypatch.setattr(retrieval, "_create_reranker", lambda model_name: FakeCrossEncoder())

    for mode in retrieval.RETRIEVAL_MODES:
        hits = retrieval.retrieve(
            "What does this paper say about hallucinations?", top_k=3,
            retrieval_mode=mode, paper="copy-of-c.pdf",
        )
        assert hits
        assert {hit["id"] for hit in hits} == {"chunk-c"}
        assert {hit["source"] for hit in hits} == {"copy-of-c.pdf"}
        assert {hit["metadata"]["source"] for hit in hits} == {"copy-of-c.pdf"}


def test_this_paper_requires_selection_and_unknown_paper_is_rejected(
    collection, monkeypatch
):
    monkeypatch.setattr(retrieval, "get_collection", lambda: collection)
    monkeypatch.setattr(retrieval, "_get_committed_snapshot", committed_snapshot)

    with pytest.raises(ValueError, match="Which paper, boss|frog requires target paper"):
        retrieval.retrieve("Summarize this paper.", retrieval_mode="dense")
    with pytest.raises(ValueError, match="not in the index"):
        retrieval.retrieve("Summarize the findings.", paper="missing.pdf")


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
    snapshot = committed_snapshot(empty)
    assert retrieval._bm25_search("anything", empty, 5, snapshot) == []
    assert retrieval._bm25_search("... !!!", empty, 5, snapshot) == []


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
    monkeypatch.setenv("RERANKER_MIN_SCORE", "0.2")
    monkeypatch.setenv("MAX_CHUNKS_PER_PAPER", "3")
    monkeypatch.setenv("MMR_LAMBDA", "0.6")
    monkeypatch.setenv("ADJACENT_CHUNKS", "2")

    config = retrieval.RetrievalConfig.from_env()

    assert (config.dense_candidates, config.bm25_candidates) == (12, 13)
    assert (config.rerank_candidates, config.final_results) == (14, 4)
    assert config.reranker_min_score == 0.2
    assert config.max_chunks_per_paper == 3
    assert config.mmr_lambda == 0.6
    assert config.adjacent_chunks == 2


@pytest.mark.parametrize("name,value", [
    ("RERANKER_MIN_SCORE", "nan"),
    ("MAX_CHUNKS_PER_PAPER", "0"),
    ("MMR_LAMBDA", "1.1"),
    ("ADJACENT_CHUNKS", "-1"),
])
def test_invalid_relevance_selection_configuration_is_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        retrieval.RetrievalConfig.from_env()


def _selection_result(chunk_id, text, score, document_id, source, page=1,
                      chunk_index=None):
    metadata = {
        "document_id": document_id,
        "file_hash": document_id,
        "source": source,
        "title": source,
        "section": "Results",
        "page": page,
    }
    if chunk_index is not None:
        metadata["chunk_index"] = chunk_index
    return {
        "id": chunk_id,
        "text": text,
        "metadata": metadata,
        "source": source,
        "title": source,
        "section": "Results",
        "page": page,
        "distance": None,
        "dense_distance": None,
        "dense_score": None,
        "bm25_score": None,
        "rrf_score": 0.01,
        "reranker_score": score,
    }


def test_threshold_removes_weak_reranker_results_and_can_return_empty():
    candidates = [
        _selection_result("strong", "direct answer", 0.8, "doc-a", "a.pdf"),
        _selection_result("weak", "unrelated text", 0.04, "doc-b", "b.pdf"),
    ]

    selected = retrieval._select_context(
        candidates, candidates, top_k=5, min_score=0.05,
        max_per_paper=2, diversity=0.75, adjacent_chunks=0,
    )
    assert [item["id"] for item in selected] == ["strong"]
    assert retrieval._select_context(
        candidates[1:], candidates, top_k=5, min_score=0.05,
        max_per_paper=2, diversity=0.75, adjacent_chunks=0,
    ) == []


def test_mmr_prefers_diverse_evidence_and_enforces_paper_cap():
    candidates = [
        _selection_result("a1", "alpha beta gamma result", 0.90, "doc-a", "a.pdf"),
        _selection_result("a2", "alpha beta gamma result repeated", 0.89, "doc-a", "a.pdf"),
        _selection_result("b1", "independent delta evidence", 0.80, "doc-b", "b.pdf"),
    ]

    diverse = retrieval._mmr_select(
        candidates, limit=2, diversity=0.4, max_per_paper=3
    )
    capped = retrieval._mmr_select(
        candidates, limit=3, diversity=1.0, max_per_paper=1
    )

    assert [item["id"] for item in diverse] == ["a1", "b1"]
    assert [item["id"] for item in capped] == ["a1", "b1"]


def test_adjacent_expansion_adds_neighbor_within_budget_and_paper_cap():
    digest = "a" * 64
    version = "b" * 16
    corpus = [
        _selection_result(
            f"{digest}-1-{index}-{version}", text, None, digest, "paper.pdf",
            chunk_index=index,
        )
        for index, text in enumerate(("before", "relevant anchor", "after"))
    ]
    anchor = {**corpus[1], "reranker_score": 0.9}

    selected = retrieval._select_context(
        [anchor], corpus, top_k=3, min_score=0.05,
        max_per_paper=2, diversity=0.75, adjacent_chunks=1,
    )

    assert [item["text"] for item in selected] == ["relevant anchor", "after"]
    assert selected[0]["selection_reason"] == "anchor"
    assert selected[1]["selection_reason"] == "adjacent"
    assert selected[1]["adjacent_to"] == anchor["id"]


def test_distinct_anchors_take_priority_over_adjacent_chunks():
    digest = "a" * 64
    corpus = [
        _selection_result(
            f"{digest}-1-{index}", text, None, digest, "paper.pdf",
            chunk_index=index,
        )
        for index, text in enumerate(("first answer", "neighbor", "second answer"))
    ]
    anchors = [
        {**corpus[0], "reranker_score": 0.9},
        {**corpus[2], "reranker_score": 0.8},
    ]

    selected = retrieval._select_context(
        anchors, corpus, top_k=5, min_score=0.01,
        max_per_paper=2, diversity=1.0, adjacent_chunks=1,
    )

    assert [item["text"] for item in selected] == ["first answer", "second answer"]
    assert all(item["selection_reason"] == "anchor" for item in selected)


def test_reranker_receives_corrected_search_question(monkeypatch):
    rows = [{
        "id": "dimensions",
        "text": "We propose six dimensions of trustworthiness. " + "dimension " * 8,
        "metadata": {"source": "survey.pdf", "title": "Survey", "section": "Abstract",
                     "page": 1, "file_hash": "survey"},
        "distance": 0.1,
    }]
    collection = FakeCollection(rows)
    monkeypatch.setattr(retrieval, "get_collection", lambda: collection)
    monkeypatch.setattr(retrieval, "_get_committed_snapshot", committed_snapshot)
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(
        retrieval, "get_index_config",
        lambda: __import__("src.index_config", fromlist=["IndexConfig"]).IndexConfig(
            embedding_dimension=2
        ),
    )
    seen_questions = []

    class FakeCrossEncoder:
        def predict(self, pairs, batch_size, show_progress_bar):
            seen_questions.extend(question for question, _ in pairs)
            return [0.8 for _ in pairs]

    monkeypatch.setattr(retrieval, "_create_reranker", lambda model_name: FakeCrossEncoder())

    hits = retrieval.retrieve(
        "What are diminention of trustworthiness?", paper="survey.pdf"
    )

    assert [item["id"] for item in hits] == ["dimensions"]
    assert seen_questions == ["What are dimension of trustworthiness?"]


def test_default_rerank_pipeline_returns_no_evidence_below_threshold(
    collection, monkeypatch
):
    monkeypatch.setattr(retrieval, "get_collection", lambda: collection)
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(
        retrieval, "get_index_config",
        lambda: __import__("src.index_config", fromlist=["IndexConfig"]).IndexConfig(
            embedding_dimension=2
        ),
    )
    monkeypatch.setattr(retrieval, "_get_committed_snapshot", committed_snapshot)

    class WeakCrossEncoder:
        def predict(self, pairs, batch_size, show_progress_bar):
            return [0.001 for _ in pairs]

    monkeypatch.setattr(retrieval, "_create_reranker", lambda model_name: WeakCrossEncoder())

    assert retrieval.retrieve(
        "unrelated question", top_k=5, retrieval_mode="hybrid-rerank"
    ) == []


def test_overview_evidence_uses_target_paper_and_its_summary_sections():
    def record(chunk_id, title, section, page, text):
        result = _selection_result(chunk_id, text, None, title, f"{title}.pdf", page)
        result["title"] = title
        result["section"] = section
        result["metadata"]["title"] = title
        result["metadata"]["section"] = section
        return result

    corpus = [
        record("survey-cover", "RAG Survey", "Untitled section", 1,
               "Authors and affiliations. " * 15),
        record("survey-abstract", "RAG Survey", "Untitled section", 1,
               "We propose a framework and review six dimensions of trustworthiness. " * 4),
        record("survey-intro", "RAG Survey", "1 Introduction", 2,
               "The introduction explains the research problem and prior work. " * 4),
        record("survey-conclusion", "RAG Survey", "6 Conclusion", 33,
               "The conclusion summarizes the study and its findings. " * 4),
        record("other-abstract", "Bias Study", "Abstract", 1,
               "We study a different subject and provide other results. " * 4),
    ]

    overview = retrieval._overview_evidence("What does the survey contribute?", corpus)

    assert [item["id"] for item in overview] == [
        "survey-abstract", "survey-conclusion", "survey-intro",
    ]
    assert all(item["selection_reason"] == "overview" for item in overview)
    assert retrieval._overview_evidence("What is quantum entanglement?", corpus) == []


def test_dimension_list_uses_first_direct_enumeration():
    abstract = _selection_result(
        "abstract", "Six dimensions of trustworthiness: factuality, robustness, "
        "fairness, transparency, accountability, and privacy.", None,
        "survey", "survey.pdf", page=1,
    )
    conclusion = _selection_result(
        "conclusion", "Six dimensions of trustworthiness: actuality, robustness, "
        "fairness, transparency, accountability, and privacy.", None,
        "survey", "survey.pdf", page=33,
    )

    selected = retrieval._overview_evidence(
        "What are dimension of trustworthiness?", [conclusion, abstract]
    )

    assert [item["id"] for item in selected] == ["abstract"]
    assert selected[0]["selection_reason"] == "direct_list"


def test_selected_paper_summary_keeps_structural_evidence_below_reranker_floor(
    monkeypatch
):
    rows = [
        {"id": "intro", "text": "We propose a framework to assess six dimensions of trust. " * 4,
         "metadata": {"source": "survey.pdf", "title": "Survey", "section": "Abstract",
                      "page": 1, "file_hash": "survey"}, "distance": 0.1},
        {"id": "conclusion", "text": "The conclusion reviews the framework and benchmark results. " * 4,
         "metadata": {"source": "survey.pdf", "title": "Survey", "section": "Conclusion",
                      "page": 10, "file_hash": "survey"}, "distance": 0.2},
    ]
    collection = FakeCollection(rows)
    monkeypatch.setattr(retrieval, "get_collection", lambda: collection)
    monkeypatch.setattr(retrieval, "_get_committed_snapshot", committed_snapshot)
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(
        retrieval, "get_index_config",
        lambda: __import__("src.index_config", fromlist=["IndexConfig"]).IndexConfig(
            embedding_dimension=2
        ),
    )

    class WeakCrossEncoder:
        def predict(self, pairs, batch_size, show_progress_bar):
            return [0.001 for _ in pairs]

    monkeypatch.setattr(retrieval, "_create_reranker", lambda model_name: WeakCrossEncoder())

    hits = retrieval.retrieve(
        "Summarize this paper.", paper="survey.pdf", retrieval_mode="hybrid-rerank"
    )
    assert {item["id"] for item in hits} == {"intro", "conclusion"}
    assert all(item["source"] == "survey.pdf" for item in hits)
    assert all(item["selection_reason"] == "overview" for item in hits)


def test_generation_context_uses_retrieved_citation_metadata(collection, monkeypatch):
    monkeypatch.setattr(retrieval, "get_collection", lambda: collection)
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(retrieval, "get_index_config", lambda: __import__("src.index_config", fromlist=["IndexConfig"]).IndexConfig(embedding_dimension=2))
    monkeypatch.setattr(retrieval, "_get_committed_snapshot", committed_snapshot)
    hit = retrieval.retrieve("RAG", top_k=1, retrieval_mode="dense")[0]

    context = build_context([hit])

    assert context.startswith("[E1]\nSource: Paper A (paper-a.pdf)\nPage: 2\nContent:")
    assert hit["metadata"]["file_hash"] == "hash-a"
