"""Shared evidence fixtures for score, allocation, and recovery checks."""
import pytest


def chunk(id='a', paper='a.pdf', text='Supported fact', score=.8, section='results'):
    return {'id': id, 'source': paper, 'title': paper, 'page': 1, 'distance': .1,
            'text': text, 'reranker_score': score, 'section': section, 'rrf_score': .01,
            'metadata': {'document_id': paper, 'document_version': 'v1'}}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for key in ('RELEVANCE_CALIBRATION', 'RERANKER_SCORE_TRANSFORM', 'PAPER_CAP_POLICY', 'EVIDENCE_RECOVERY', 'GENERATION_TOTAL_CALLS'):
        monkeypatch.delenv(key, raising=False)


