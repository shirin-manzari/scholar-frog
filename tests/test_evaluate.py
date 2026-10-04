import json

import pytest

from src.citations import CitationValidation, Evidence, GenerationResult, GenerationStatus
from src.evaluate import evaluate_dataset, load_dataset


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps({"version": 1, "name": "test", "cases": [
        {"id": "supported", "question": "Supported?", "expected": "answered",
         "evidence": [{"source": "paper.pdf", "page": 2}]},
        {"id": "abstain", "question": "Unknown?", "expected": "abstained", "evidence": []},
    ]}))
    return path


def test_dataset_validation_rejects_answered_case_without_evidence(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"version": 1, "cases": [
        {"id": "bad", "question": "Q", "expected": "answered", "evidence": []},
    ]}))
    with pytest.raises(ValueError, match="needs at least one"):
        load_dataset(path)


def test_retrieval_metrics_and_per_paper_coverage(dataset):
    def fake_retrieve(question, **kwargs):
        return ([{"id": "correct", "source": "paper.pdf", "page": 2}]
                if question == "Supported?" else [])

    report = evaluate_dataset(dataset, retrieve_fn=fake_retrieve)

    assert report["metrics"]["retrieval_recall_at_5"] == 1.0
    assert report["metrics"]["per_paper_coverage"] == {
        "paper.pdf": {"expected_cases": 1, "retrieval_hits": 1, "recall": 1.0}
    }
    assert report["cases"][1]["retrieval_hit"] is False


def test_generation_metrics_track_citation_locator_and_abstention(dataset):
    def fake_retrieve(question, **kwargs):
        return [{"id": "correct", "source": "paper.pdf", "page": 2}]

    evidence = Evidence("E1", "correct", "hash", "Paper", "paper.pdf", 2, "Support")
    answered_validation = CitationValidation(True, ["E1"], [], [], [evidence])
    abstained_validation = CitationValidation(False, [], [], [], [], applicable=False)

    def fake_generate(question, chunks):
        if question == "Supported?":
            return GenerationResult("Answer [E1].", "", [evidence], answered_validation, 0)
        return GenerationResult("No answer", "", [], abstained_validation, 0,
                                status=GenerationStatus.ABSTAINED)

    report = evaluate_dataset(dataset, include_generation=True,
                              retrieve_fn=fake_retrieve, generate_fn=fake_generate)

    metrics = report["metrics"]
    assert metrics["generation_outcome_accuracy"] == 1.0
    assert metrics["abstention_correctness"] == 1.0
    assert metrics["citation_locator_precision"] == 1.0
    assert metrics["citation_valid_answer_rate"] == 1.0


def test_anchor_page_recall_is_separate_from_expanded_context_page_recall(dataset):
    def fake_retrieve(question, **kwargs):
        return [{'id': 'group', 'source': 'paper.pdf', 'page': 2,
                 'anchor_hits': [{'id': 'anchor', 'page': 1}],
                 'metadata': {'context_usage': {'anchor_count': 1, 'evidence_tokens': 42}}}]
    report = evaluate_dataset(dataset, retrieve_fn=fake_retrieve)
    assert report['metrics']['anchor_page_recall'] == 0.0
    assert report['metrics']['context_page_recall'] == 1.0
    assert report['cases'][0]['selected_anchor_count'] == 1
    assert report['cases'][0]['context_usage']['evidence_tokens'] == 42


def test_evaluation_reports_effective_query_and_context_settings(dataset, monkeypatch):
    monkeypatch.setenv('QUERY_INSTRUCTION', 'off')
    monkeypatch.setenv('QUERY_EXPANSION', 'true')
    monkeypatch.setenv('QUERY_MAX_VARIANTS', '1')
    monkeypatch.setenv('DENSE_CANDIDATES', '17')
    monkeypatch.setenv('CONTEXT_PARAGRAPHS', '0')
    report = evaluate_dataset(dataset, retrieval_mode='hybrid', retrieve_fn=lambda *a, **kw: [])
    settings = report['effective_settings']
    assert settings['query_instruction'] == ''
    assert settings['query_expansion'] is True
    assert settings['query_max_variants'] == 1
    assert settings['retrieval']['dense_candidates'] == 17
    assert settings['retrieval']['mode'] == 'hybrid'
    assert settings['context']['CONTEXT_PARAGRAPHS'] == 0
    assert report['metrics']['context_locator_mrr'] == 0
    assert report['metrics']['negative_cases_with_evidence'] == 0


def test_anchor_mrr_uses_anchor_rank_not_context_block_order(dataset):
    def provider(question, **kwargs):
        return [{'source': 'paper.pdf', 'page': 2, 'text': 'Reviewed excerpt',
                 'anchor_hits': [{'id': 'hit', 'page': 2, 'rank': 3}]}]
    report = evaluate_dataset(dataset, retrieve_fn=provider, include_evidence_text=True)
    assert report['metrics']['context_locator_mrr'] == 1.0
    assert report['metrics']['anchor_locator_mrr'] == .3333
    assert report['metrics']['negative_evidence_blocks'] == 1
    assert report['cases'][0]['retrieved'][0]['text'] == 'Reviewed excerpt'
