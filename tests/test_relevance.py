"""Model-specific score contracts and offline calibration tradeoffs."""
import json
from types import SimpleNamespace

import pytest
from src import retrieve as r
from src.relevance import calibrate, effective_threshold, score_contract, threshold_metrics
from evidence_helpers import chunk, clean


def test_score_contract_actual_activation_and_probability():
    model = SimpleNamespace(activation_fn=type('Sigmoid', (), {})(), num_labels=1,
                            config=SimpleNamespace(_commit_hash='revision'))
    contract = score_contract(model, 'model')
    assert contract['effective_activation'] == 'Sigmoid'
    assert contract['revision'] == 'revision'
    assert not contract['score_is_probability']


@pytest.mark.parametrize('transform,expected', [('raw', 'Identity'), ('sigmoid', 'Sigmoid')])
def test_explicit_score_activation(monkeypatch, transform, expected):
    monkeypatch.setenv('RERANKER_SCORE_TRANSFORM', transform)
    monkeypatch.setenv('RERANKER_MIN_SCORE', '.01')
    calls = []
    class Model:
        num_labels = 1
        def predict(self, pairs, **kwargs):
            calls.append(type(kwargs['activation_fn']).__name__)
            return [.5]
    monkeypatch.setattr(r, '_get_reranker', lambda name: Model())
    result = r._rerank('question', [chunk()], 'model')
    assert calls == [expected]
    assert result[0]['score_contract']['transformation'] == transform


@pytest.mark.parametrize('key', ['model', 'revision', 'transformation', 'effective_activation'])
def test_incompatible_calibration(monkeypatch, tmp_path, key):
    contract = {'model': 'model', 'revision': 'r', 'transformation': 'raw', 'effective_activation': 'raw'}
    path = tmp_path / 'cal.json'
    path.write_text(json.dumps({'status': 'human-reviewed', 'score_contract': {**contract, key: 'other'},
                               'held_out': [1], 'calibration': [{'threshold': .2}], 'selected_threshold': .2}))
    monkeypatch.setenv('RELEVANCE_CALIBRATION', str(path))
    with pytest.raises(ValueError, match=key):
        effective_threshold(contract, .01)


def test_threshold_boundary_and_both_error_directions():
    data = [{'score': .5, 'relevant': True}, {'score': .49, 'relevant': True},
            {'score': .6, 'relevant': False}, {'score': .1, 'relevant': False}]
    metrics = threshold_metrics(data, .5)
    assert metrics == {'threshold': .5, 'tp': 1, 'fp': 1, 'fn': 1, 'tn': 1,
                       'precision': .5, 'recall': .5, 'false_rejection_rate': .5, 'false_acceptance_rate': .5}
    hits = r._select_context([chunk('a', score=.5), chunk('b', score=.499)], top_k=4,
                            min_score=.5, max_per_paper=2, diversity=.75)
    assert [h['id'] for h in hits] == ['a']


def labels():
    return [{'query': f'{split}-{kind}', 'passage': kind, 'split': split, 'kind': kind,
             'relevant': kind == 'relevant', 'annotator': 'human'}
            for split in ('calibration', 'held_out') for kind in ('relevant', 'topical_negative', 'out_of_corpus')]


def test_calibration_separation_and_reports(tmp_path):
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps({'examples': labels()}))
    result = calibrate(path, [.1, .5], .5, lambda pairs: [.7, .6, .01], {})
    assert result['calibration'][1]['precision'] == .5
    assert result['held_out'][1]['recall'] == 1
    data = labels(); data[3]['query'] = data[0]['query']; data[3]['passage'] = 'distinct passage'
    path.write_text(json.dumps({'examples': data}))
    with pytest.raises(ValueError, match='leakage'):
        calibrate(path, [.5], .5, lambda pairs: [], {})


def test_compatible_threshold_is_used(monkeypatch, tmp_path):
    contract = {'model': 'model', 'revision': 'r', 'transformation': 'raw', 'effective_activation': 'raw'}
    artifact = {'status': 'human-reviewed', 'score_contract': contract, 'selected_threshold': .5,
                'calibration': [{'threshold': .5}], 'held_out': [{'threshold': .5}]}
    path = tmp_path / 'calibration.json'; path.write_text(json.dumps(artifact))
    monkeypatch.setenv('RELEVANCE_CALIBRATION', str(path))
    assert effective_threshold(contract, .01) == .5
    artifact['selected_threshold'] = .6; path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match='not evaluated'):
        effective_threshold(contract, .01)


@pytest.mark.parametrize('transform,expected', [('raw', -2.0), ('sigmoid', 0.11920292)])
def test_transform_controls_numeric_scores(monkeypatch, transform, expected):
    import torch
    monkeypatch.setenv('RERANKER_SCORE_TRANSFORM', transform)
    monkeypatch.setenv('RERANKER_MIN_SCORE', '.01')
    class Model:
        num_labels = 1
        def predict(self, pairs, **kwargs):
            return [kwargs['activation_fn'](torch.tensor(-2.0)).item()]
    monkeypatch.setattr(r, '_get_reranker', lambda name: Model())
    assert r._rerank('Q', [chunk()], 'model')[0]['reranker_score'] == pytest.approx(expected)


def test_multiclass_and_nonfinite_scores_fail_explicitly(monkeypatch):
    with pytest.raises(ValueError, match='single-output'):
        score_contract(SimpleNamespace(num_labels=2), 'model')
    monkeypatch.setattr(r, '_get_reranker', lambda name: SimpleNamespace(predict=lambda *a, **kw: [float('nan')]))
    with pytest.raises(ValueError, match='invalid scalar scores'):
        r._rerank('Q', [chunk()], 'model')


@pytest.mark.parametrize('setting,value', [('RERANKER_MODEL', 'different-model'), ('RERANKER_SCORE_TRANSFORM', 'raw')])
def test_implicit_baseline_threshold_is_model_specific(monkeypatch, setting, value):
    monkeypatch.delenv('RERANKER_MIN_SCORE', raising=False)
    monkeypatch.setenv(setting, value)
    with pytest.raises(ValueError, match='model-specific'):
        r.RetrievalConfig.from_env()
    monkeypatch.setenv('RERANKER_MIN_SCORE', '.1')
    assert r.RetrievalConfig.from_env().reranker_min_score == .1


def test_sigmoid_threshold_scale_is_enforced():
    with pytest.raises(ValueError, match='sigmoid'):
        effective_threshold({'effective_activation': 'Sigmoid'}, 2.0)
