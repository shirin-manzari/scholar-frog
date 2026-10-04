"""Bounded recovery, stable citations, and shared evidence/call budgets."""
import json

import pytest
from src import generate as g, retrieve as r
from src.citations import GenerationStatus
from src.recovery import combine_evidence, recovery_query
from evidence_helpers import chunk, clean


def test_recovery_query_preserves_original_and_does_not_assert_claim():
    question = 'Does Nguyen report no benefit at 20 mg/L in 2025?'
    query = recovery_query(question, [{'supported': False, 'claim': 'Benefit is 50%.'}])
    assert query.startswith(question)
    assert 'not an established fact' in query
    assert recovery_query(question, []) is None


def test_combine_dedup_stable_ids_and_budgets(monkeypatch):
    from src.citations import assign_evidence
    original = [chunk('a')]
    combined = combine_evidence('Q', original, [chunk('a'), chunk('b', text='Different fact')], 2)
    assert [e.chunk_id for e in assign_evidence(combined)] == ['a', 'b']
    assert [e.evidence_id for e in assign_evidence(combined)] == ['E1', 'E2']
    assert combine_evidence('Q', original, [chunk('b')], 1) == original
    monkeypatch.setenv('CONTEXT_BUDGET_TOKENS', '1')
    assert combine_evidence('Q', original, [chunk('b')], 2) == original


def test_overlap_version_and_expansion_budget(monkeypatch):
    a = chunk('a', text='abcdef'); a['metadata'].update(character_start=0, character_end=6)
    b = chunk('b', text='defghi'); b['metadata'].update(character_start=3, character_end=9)
    assert combine_evidence('Q', [a], [b], 3) == [a]
    b['metadata']['document_version'] = 'v2'
    assert len(combine_evidence('Q', [a], [b], 3)) == 2
    b['evidence_kind'] = 'expansion'
    monkeypatch.setenv('CONTEXT_EXPANSION_TOKENS', '0')
    assert combine_evidence('Q', [a], [b], 3) == [a]


def answered(text='Fact [E1].'):
    return json.dumps({'status': 'answered', 'answer': text})


def verdict(supported):
    return json.dumps({'verdicts': [{'claim_id': 'C1', 'supported': supported, 'reason': 'missing conditions' if not supported else 'supported'}]})


def run(monkeypatch, responses, recovery, retries=0):
    monkeypatch.setenv('EVIDENCE_RECOVERY', 'true')
    calls = []
    seq = iter(responses)
    monkeypatch.setattr(g, '_call_backend', lambda *args: calls.append(args) or next(seq))
    result = g.generate_answer('What fact in a.pdf?', [chunk()], max_retries=retries, recovery_fn=recovery)
    return result, calls


def test_successful_recovery_one_round_and_revalidation(monkeypatch):
    queries = []
    result, calls = run(monkeypatch, [answered(), verdict(False), answered('New fact [E2].'), verdict(True)],
                        lambda q: queries.append(q) or [chunk('b', text='New fact')])
    assert result.status is GenerationStatus.ANSWERED
    assert len(queries) == 1
    assert len(calls) == 4
    assert result.debug['recovery_success']
    assert result.evidence[0].evidence_id == 'E1'
    assert result.evidence[1].evidence_id == 'E2'


@pytest.mark.parametrize('responses', [
    ['not json'], [answered('Fact [E99].')], [answered(), 'bad verifier json'],
])
def test_non_actionable_failures_do_not_search(monkeypatch, responses):
    result, calls = run(monkeypatch, responses, lambda q: pytest.fail('Should not search'))
    assert result.debug['recovery_rounds'] == 0
    assert result.status is GenerationStatus.VALIDATION_FAILED


def test_verifier_outage_does_not_search(monkeypatch):
    monkeypatch.setenv('EVIDENCE_RECOVERY', 'true')
    calls = []
    def backend(*args):
        calls.append(args)
        if len(calls) == 1:
            return answered()
        raise RuntimeError('outage')
    monkeypatch.setattr(g, '_call_backend', backend)
    result = g.generate_answer('Q', [chunk()], max_retries=0, recovery_fn=lambda q: pytest.fail('No search'))
    assert result.debug['recovery_rounds'] == 0


def test_final_abstention_and_strict_one_round(monkeypatch):
    searches = []
    result, calls = run(monkeypatch, [answered(), verdict(False), answered(), verdict(False)],
                        lambda q: searches.append(q) or [])
    assert len(searches) == 1
    assert len(calls) == 4
    assert result.status is GenerationStatus.ABSTAINED
    assert 'Fact' not in result.answer


def test_supported_partial_without_new_evidence(monkeypatch):
    result, calls = run(monkeypatch, [answered(), verdict(False), answered('Supported part [E1].'), verdict(True)], lambda q: [])
    assert result.status is GenerationStatus.ANSWERED
    assert not result.debug['recovery_success']
    assert 'Answer only supported parts' in calls[-2][-1]


def test_call_budget_includes_retries_and_verifier(monkeypatch):
    monkeypatch.setenv('GENERATION_TOTAL_CALLS', '4')
    result, calls = run(monkeypatch, [answered(), verdict(False), answered(), verdict(False)],
                        lambda q: pytest.fail('Budget has no space for recovery'), retries=1)
    assert result.debug['model_calls'] == 4
    assert result.debug['recovery_rounds'] == 0


def test_missing_fact_recovery_and_scope(monkeypatch):
    monkeypatch.setenv('EVIDENCE_RECOVERY', 'true')
    seq = iter([json.dumps({'status': 'abstained', 'reason': 'insufficient_evidence', 'answer': '', 'missing_fact': 'What fact'}), answered(), verdict(True)])
    monkeypatch.setattr(g, '_call_backend', lambda *a: next(seq))
    original = chunk()
    original['metadata']['retrieval_scope'] = {'plan': {'task': 'focused', 'papers': ('a.pdf',), 'dimensions': (), 'positions': ()},
                                               'anchor_budget': 2, 'mode': 'hybrid-rerank', 'committed_revision': 'snapshot'}
    searches = []
    def search(q, **kwargs):
        searches.append((q, kwargs))
        kwargs['debug']['committed_revision'] = 'snapshot'
        return []
    monkeypatch.setattr(r, 'retrieve', search)
    result = g.generate_answer('What fact in a.pdf?', [original], max_retries=0)
    assert searches[0][1]['plan'].papers == ('a.pdf',)
    assert result.debug['recovery_reason'] == 'missing_fact'
    assert result.debug['model_calls'] == 3


def test_unanchored_missing_fact_is_invalid(monkeypatch):
    result, _ = run(monkeypatch, [json.dumps({'status': 'abstained', 'reason': 'insufficient_evidence', 'answer': '', 'missing_fact': 'Invented result'})], lambda q: pytest.fail('No search'))
    assert result.status is GenerationStatus.VALIDATION_FAILED


def test_recovery_is_disabled_without_semantic_validation(monkeypatch):
    monkeypatch.setenv('EVIDENCE_RECOVERY', 'true')
    monkeypatch.setattr(g, '_call_backend', lambda *a: json.dumps({'status': 'abstained', 'reason': 'insufficient_evidence', 'answer': '', 'missing_fact': 'What fact'}))
    result = g.generate_answer('What fact?', [chunk()], max_retries=0, semantic_validation_enabled=False,
                               recovery_fn=lambda q: pytest.fail('Cannot recover without semantic validation'))
    assert result.debug['recovery_rounds'] == 0


def test_verifier_budget_exhaustion_fails_closed(monkeypatch):
    monkeypatch.setenv('GENERATION_TOTAL_CALLS', '1')
    result, calls = run(monkeypatch, [answered()], lambda q: pytest.fail('No budget'), retries=0)
    assert len(calls) == 1
    assert result.status is GenerationStatus.VALIDATION_FAILED
    assert 'exhausted' in result.validation.errors[0]


def test_last_malformed_response_does_not_reuse_stale_gap(monkeypatch):
    result, calls = run(monkeypatch, [answered(), verdict(False), 'invalid'], lambda q: pytest.fail('Stale gap'), retries=1)
    assert result.debug['recovery_rounds'] == 0


def test_snapshot_change_excludes_recovery_evidence(monkeypatch):
    monkeypatch.setenv('EVIDENCE_RECOVERY', 'true')
    responses = iter([answered(), verdict(False), answered(), verdict(True)])
    monkeypatch.setattr(g, '_call_backend', lambda *a: next(responses))
    original = chunk()
    original['metadata']['retrieval_scope'] = {'plan': {'papers': ('a.pdf',)}, 'mode': 'hybrid-rerank',
                                               'anchor_budget': 3, 'committed_revision': 'old'}
    def search(q, **kw):
        kw['debug']['committed_revision'] = 'new'
        return [chunk('b', text='New')]
    monkeypatch.setattr(r, 'retrieve', search)
    result = g.generate_answer('Q', [original], max_retries=0)
    assert result.debug['new_evidence_count'] == 0
    assert 'snapshot changed' in result.debug['recovery_error']


def test_actionable_gap_abstains_if_call_budget_prevents_recovery(monkeypatch):
    monkeypatch.setenv('GENERATION_TOTAL_CALLS', '2')
    result, calls = run(monkeypatch, [answered(), verdict(False)], lambda q: pytest.fail('No budget'))
    assert result.status is GenerationStatus.ABSTAINED
    assert result.debug['recovery_skipped_reason'] == 'total_call_budget'
    assert len(calls) == 2


def test_passage_paper_caps_ignore_context_only_blocks():
    a = chunk('a'); a['anchor_hits'] = [{'id': 'a'}]
    expansion = chunk('ctx', text='Context only'); expansion['anchor_hits'] = []
    b = chunk('b', text='Additional anchor'); b['anchor_hits'] = [{'id': 'b'}]
    combined = combine_evidence('Q', [a, expansion], [b], 2)
    assert len(combined) == 3


