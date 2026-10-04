"""Offline labels -> threshold tradeoffs. The operator explicitly selects a threshold.

python -m evaluations.calibrate_relevance labels.json report.json --thresholds .001 .01 .05 --select .01
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labels'); parser.add_argument('report')
    parser.add_argument('--thresholds', type=float, nargs='+', required=True)
    parser.add_argument('--select', type=float, required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from src.retrieve import _get_reranker
    from src.relevance import calibrate, score_contract
    from torch import nn
    name = os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-base').strip()
    if not name:
        parser.error('RERANKER_MODEL must not be empty')
    model = _get_reranker(name)
    contract = score_contract(model, name)
    kwargs = {} if contract['transformation'] == 'model-default' else {
        'activation_fn': nn.Identity() if contract['transformation'] == 'raw' else nn.Sigmoid()}
    score = lambda pairs: model.predict(pairs, batch_size=16, show_progress_bar=False, **kwargs)
    baseline = ([float(os.getenv('RERANKER_MIN_SCORE', '.01'))]
                if 'RERANKER_MIN_SCORE' in os.environ or (name == 'BAAI/bge-reranker-base'
                    and contract['transformation'] == 'model-default') else [])
    report = calibrate(args.labels, args.thresholds + baseline, args.select, score, contract)
    Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('score_contract', 'calibration', 'held_out')}, indent=2))


if __name__ == '__main__':
    main()
