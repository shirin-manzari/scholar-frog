"""Model-specific score contracts and offline, human-labeled threshold analysis."""
import hashlib
import json
import math
import os
from importlib.metadata import version
from pathlib import Path


def score_contract(model, name):
    activation = getattr(model, 'activation_fn', getattr(model, 'default_activation_function', None))
    default = type(activation).__name__ if activation is not None else 'unknown'
    transformation = os.getenv('RERANKER_SCORE_TRANSFORM', 'model-default')
    if transformation not in {'model-default', 'raw', 'sigmoid'}:
        raise ValueError('RERANKER_SCORE_TRANSFORM must be model-default, raw or sigmoid')
    config = getattr(model, 'config', None)
    labels = getattr(model, 'num_labels', getattr(config, 'num_labels', 1))
    if labels != 1:
        raise ValueError('Relevance thresholds require a single-output reranker')
    return {'wrapper_version': version('sentence-transformers'), 'model': name, 'revision': getattr(config, '_commit_hash', None),
            'transformation': transformation, 'default_activation': default,
            'effective_activation': default if transformation == 'model-default' else transformation,
            'score_scale': ('[0,1]' if transformation == 'sigmoid' or transformation == 'model-default' and default == 'Sigmoid'
                            else 'unbounded logits' if transformation == 'raw' else 'model-defined'),
            'score_is_probability': False}


def effective_threshold(contract, baseline):
    path = os.getenv('RELEVANCE_CALIBRATION', '').strip()
    if not path:
        if contract['effective_activation'] in {'Sigmoid', 'sigmoid'} and not 0 <= baseline <= 1:
            raise ValueError('Incompatible threshold: sigmoid score thresholds must lie in [0,1]')
        return baseline
    artifact = json.loads(Path(path).read_text())
    if artifact.get('status') != 'human-reviewed' or not artifact.get('held_out'):
        raise ValueError('Calibration requires human-reviewed labels and held-out results')
    for key in ('model', 'revision', 'transformation', 'effective_activation'):
        if artifact.get('score_contract', {}).get(key) != contract[key]:
            raise ValueError(f'Incompatible relevance calibration: {key}')
    if contract['revision'] is None:
        raise ValueError('Calibrated thresholds require an identifiable model revision')
    threshold = float(artifact['selected_threshold'])
    if contract['effective_activation'] in {'Sigmoid', 'sigmoid'} and not 0 <= threshold <= 1:
        raise ValueError('Incompatible calibration threshold: sigmoid scale is [0,1]')
    if not math.isfinite(threshold):
        raise ValueError('Calibration threshold must be finite')
    if threshold not in [row['threshold'] for row in artifact['calibration']] or threshold not in [row['threshold'] for row in artifact['held_out']]:
        raise ValueError('Selected threshold was not evaluated')
    return threshold


def threshold_metrics(examples, threshold):
    tp = fp = fn = tn = 0
    for row in examples:
        accepted = row['score'] >= threshold
        tp += accepted and row['relevant']
        fp += accepted and not row['relevant']
        fn += not accepted and row['relevant']
        tn += not accepted and not row['relevant']
    rate = lambda a, b: a / b if b else None
    return {'threshold': threshold, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': rate(tp, tp + fp), 'recall': rate(tp, tp + fn),
            'false_rejection_rate': rate(fn, tp + fn), 'false_acceptance_rate': rate(fp, fp + tn)}


def calibrate(path, thresholds, selected_threshold, score_fn, contract):
    """Score fixed passages locally; never choose a threshold from held-out labels."""
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    rows = data['examples']
    groups = {'calibration': [], 'held_out': []}
    queries = {key: set() for key in groups}
    seen = set()
    for row in rows:
        if (not isinstance(row, dict) or type(row.get('relevant')) is not bool or row.get('split') not in groups
                or not isinstance(row.get('annotator'), str) or not row['annotator'].strip()
                or not isinstance(row.get('query'), str) or not row['query'].strip()
                or not isinstance(row.get('passage'), str) or not row['passage'].strip()
                or row.get('kind') not in
                {'relevant', 'topical_negative', 'out_of_corpus'}):
            raise ValueError('Each example needs manual boolean label, annotator, kind and split')
        if row['relevant'] != (row['kind'] == 'relevant'):
            raise ValueError('Relevance label and example kind disagree')
        key = (row['query'], row['passage'])
        if key in seen:
            raise ValueError('Duplicate query/passage label')
        seen.add(key)
        queries[row['split']].add(' '.join(row['query'].split()).casefold())
        groups[row['split']].append(row)
    if queries['calibration'] & queries['held_out']:
        raise ValueError('Query leakage across calibration and held-out splits')
    for split, examples in groups.items():
        if {r['kind'] for r in examples} != {'relevant', 'topical_negative', 'out_of_corpus'}:
            raise ValueError(f'{split} needs relevant, topical-negative and out-of-corpus labels')
        scores = list(score_fn([(r['query'], r['passage']) for r in examples]))
        if len(scores) != len(examples):
            raise ValueError('Score count mismatch')
        for row, score in zip(examples, scores):
            row['score'] = float(score)
            if not math.isfinite(row['score']):
                raise ValueError('Non-finite relevance score')
    thresholds = sorted(set(float(t) for t in thresholds))
    if selected_threshold not in thresholds or not all(math.isfinite(t) for t in thresholds):
        raise ValueError('Explicit selected threshold must be among finite candidates')
    return {'status': 'human-reviewed', 'dataset_sha256': hashlib.sha256(raw).hexdigest(),
            'score_contract': contract, 'snapshot_revision': data.get('snapshot_revision'),
            'selected_threshold': selected_threshold,
            'selection_policy': 'Explicit operator choice; inspect precision/recall and both error rates',
            **{split: [threshold_metrics(examples, t) for t in thresholds]
               for split, examples in groups.items()}, 'scored_examples': rows}
