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
