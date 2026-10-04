"""Task-aware allocation over relevant, whole anchors."""
import pytest
from src import retrieve as r
from src.evidence_policy import RetrievalPlan, select_anchors
from evidence_helpers import chunk, clean


def test_focused_single_paper_uses_anchor_budget():
    hits = [chunk(str(i)) for i in range(5)]
    assert len(select_anchors(hits, RetrievalPlan(papers=('a.pdf',)), 5, 2, r._paper_key)) == 5
    hits += [chunk('b', 'b.pdf')]
    assert len(select_anchors(hits, RetrievalPlan(), 5, 2, r._paper_key)) == 3


def test_summary_covers_sections_before_repetition():
    hits = [chunk('1'), chunk('2'), chunk('3', section='methods'), chunk('4', section='conclusion')]
    chosen = select_anchors(hits, RetrievalPlan('summary', ('a.pdf',)), 3, 2, r._paper_key)
    assert {h['section'] for h in chosen} == {'results', 'methods', 'conclusion'}


def test_comparison_fairness_missing_dimensions_and_redistribution():
    hits = [chunk('a1', text='accuracy conditions'), chunk('a2', text='latency'),
            chunk('a3', text='accuracy'), chunk('b1', 'b.pdf', text='accuracy')]
    debug = {}
    chosen = select_anchors(hits, RetrievalPlan('comparison', ('a.pdf', 'b.pdf'), ('accuracy', 'latency')),
                            4, 2, r._paper_key, debug)
    assert [c['id'] for c in chosen[:2]] == ['a1', 'b1']
    assert {'paper': 'b.pdf', 'dimension': 'latency', 'reason': 'no_relevant_evidence'} in debug['missing_allocations']
    assert debug['paper_allocations'] == {'a.pdf': 3, 'b.pdf': 1}


def test_no_quota_filling_and_conflict_conditions():
    plan = RetrievalPlan('conflict', ('a.pdf', 'b.pdf'), positions=('benefit adults', 'no benefit children'))
    debug = {}
    chosen = select_anchors([chunk('a', text='benefit adults')], plan, 5, 2, r._paper_key, debug)
    assert len(chosen) == 1
    assert any(gap['paper'] == 'b.pdf' for gap in debug['missing_allocations'])


def test_pending_exclusion_under_task_aware(monkeypatch):
    from test_query_processing import setup_pipeline, record
    from test_retrieve import committed_snapshot
    from src.sync import CommittedSnapshot
    c, _ = setup_pipeline(monkeypatch, [record('Relevant', chunk='committed'), record('Relevant', chunk='pending')])
    snapshot = committed_snapshot(c)
    snapshot = CommittedSnapshot('revision', frozenset({'committed'}), {'committed': snapshot.owners['committed']})
    monkeypatch.setattr(r, '_get_committed_snapshot', lambda c: snapshot)
    monkeypatch.setattr(r, '_rerank', lambda q, hits, name: [dict(h, reranker_score=.8) for h in hits])
    monkeypatch.setenv('PAPER_CAP_POLICY', 'task-aware')
    assert [h['id'] for h in r.retrieve('Relevant')] == ['committed']


def test_context_admission_preserves_fair_whole_passages():
    hits = [chunk('large-a', text='x' * 100), chunk('small-a', text='fact'), chunk('small-b', 'b.pdf', text='fact')]
    debug = {}
    chosen = select_anchors(hits, RetrievalPlan('comparison', ('a.pdf', 'b.pdf')), 5, 2,
                            r._paper_key, debug, fits=lambda rows: sum(len(c['text']) for c in rows) <= 8)
    assert [c['id'] for c in chosen] == ['small-a', 'small-b']
    assert {'id': 'large-a', 'reason': 'context_budget'} in debug['budget_rejected_candidates']


def test_multiple_paper_scope_excludes_other_papers(monkeypatch):
    from test_query_processing import setup_pipeline, record
    c, _ = setup_pipeline(monkeypatch, [record('Relevant', 'a', 'a'), record('Relevant', 'b', 'b'), record('Relevant', 'c', 'c')])
    original_query = c.query
    def query(**kwargs):
        where = kwargs.pop('where', None)
        results = original_query(**kwargs)
        if where:
            ids = set(where['document_id']['$in'])
            keep = [i for i, meta in enumerate(results['metadatas'][0]) if meta['document_id'] in ids]
            results = {key: [[values[0][i] for i in keep]] for key, values in results.items()}
        return results
    monkeypatch.setattr(c, 'query', query)
    monkeypatch.setattr(r, '_rerank', lambda q, hits, name: [dict(h, reranker_score=.8) for h in hits])
    monkeypatch.setenv('PAPER_CAP_POLICY', 'task-aware')
    plan = RetrievalPlan('comparison', ('a.pdf', 'b.pdf'))
    assert {h['source'] for h in r.retrieve('Compare Relevant', plan=plan)} == {'a.pdf', 'b.pdf'}


def test_structural_summary_must_pass_relevance_in_task_policy(monkeypatch):
    from test_query_processing import setup_pipeline, record
    c, _ = setup_pipeline(monkeypatch, [record('We propose an important research contribution. ' * 8)])
    monkeypatch.setenv('PAPER_CAP_POLICY', 'task-aware')
    monkeypatch.setattr(r, '_rerank', lambda q, hits, name: [dict(h, reranker_score=.001) for h in hits])
    assert r.retrieve('Summarize this paper.', paper='a.pdf') == []


def test_conflict_preserves_both_positions_and_conditions():
    plan = RetrievalPlan('conflict', ('a.pdf', 'b.pdf'), positions=('benefit adults', 'no benefit children'))
    hits = [chunk('a', text='benefit adults at 20 mg/L'), chunk('b', 'b.pdf', text='no benefit children at 10 mg/L')]
    debug = {}
    chosen = select_anchors(hits, plan, 2, 2, r._paper_key, debug)
    assert {c['id'] for c in chosen} == {'a', 'b'}
    assert '20 mg/L' in chosen[0]['text'] and '10 mg/L' in chosen[1]['text']


def test_negated_conflict_position_does_not_fill_positive_position():
    plan = RetrievalPlan('conflict', ('a.pdf',), positions=('benefit adults', 'no benefit adults'))
    debug = {}
    select_anchors([chunk(text='No benefit adults at 20 mg/L.')], plan, 3, 2, r._paper_key, debug)
    assert debug['dimension_coverage']['a.pdf']['benefit adults'] == 0
    assert debug['dimension_coverage']['a.pdf']['no benefit adults'] == 1
    assert {'paper': 'a.pdf', 'dimension': 'benefit adults', 'reason': 'no_relevant_evidence'} in debug['missing_allocations']


def test_condition_numbers_are_matched_as_complete_tokens():
    debug = {}
    select_anchors([chunk(text='A dose of 200 mg/L.')], RetrievalPlan('comparison', ('a.pdf',), ('20 mg/L',)),
                   3, 2, r._paper_key, debug)
    assert debug['dimension_coverage']['a.pdf']['20 mg/L'] == 0


