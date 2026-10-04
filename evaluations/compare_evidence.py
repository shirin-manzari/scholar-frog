"""Isolated snapshot evaluation; calibration and generation are explicitly gated.

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m evaluations.compare_evidence
Supply --calibration PATH for the separately measured calibrated threshold arm.
Use --generation only with a working, explicitly configured generation backend.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='evaluations/query-golden-v1.json')
    parser.add_argument('--output', default='evaluations/evidence-comparison.json')
    parser.add_argument('--calibration')
    parser.add_argument('--generation', action='store_true')
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from src import ingest, retrieve, sync
    from src.evaluate import evaluate_dataset
    from src.relevance import score_contract
    original_db = ingest.DB_DIR
    # Freeze policy settings independently of the user's .env; index config is preserved.
    settings = {'DENSE_CANDIDATES': '50', 'BM25_CANDIDATES': '50', 'RERANK_CANDIDATES': '40',
                'FINAL_RESULTS': '5', 'RRF_K': '60', 'RERANKER_MODEL': 'BAAI/bge-reranker-base',
                'RERANKER_SCORE_TRANSFORM': 'model-default', 'RERANKER_MIN_SCORE': '.01',
                'MAX_CHUNKS_PER_PAPER': '2', 'MMR_LAMBDA': '.75', 'QUERY_INSTRUCTION': 'auto',
                'QUERY_EXPANSION': 'false', 'QUERY_MAX_VARIANTS': '2',
                'PAPER_CAP_POLICY': 'baseline', 'EVIDENCE_RECOVERY': 'false', 'RELEVANCE_CALIBRATION': '',
                'CONTEXT_BUDGET_TOKENS': '4000', 'CONTEXT_WINDOW_TOKENS': '8192',
                'CONTEXT_PARAGRAPHS': '1', 'CONTEXT_EXPANSION_TOKENS': '1500',
                'CONTEXT_SAME_SECTION': 'true', 'CONTEXT_ANSWER_RESERVE': '1024',
                'CONTEXT_FRAMING_RESERVE': '32', 'CONTEXT_RETRY_RESERVE': '256',
                'CITATION_MAX_RETRIES': '1', 'CITATION_SEMANTIC_VALIDATION': 'true',
                'GENERATION_TOTAL_CALLS': '6'}
    previous = {k: os.environ.get(k) for k in settings}
    report = {'dataset_sha256': hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
              'configurations': {}, 'limitations': [
                  'Existing golden labels are source/page proxies, not complete passage relevance judgments.',
                  'No blind human answer review: answered status is not factual correctness.',
                  'Small focused-query set; summary, comparison and conflict generalization needs labeled cases.']}
    try:
        os.environ.update(settings)
        with tempfile.TemporaryDirectory(prefix='scholar-frog-evidence-') as root:
            with sync.INDEX_LOCK:
                shutil.copytree(original_db, Path(root) / 'index')
            ingest.DB_DIR = str(Path(root) / 'index')
            collection = ingest.get_collection()
            snapshot = retrieve._get_committed_snapshot(collection)
            def digest():
                rows = collection.get(ids=sorted(snapshot.chunk_ids), include=['documents', 'metadatas', 'embeddings'])
                rows['embeddings'] = rows['embeddings'].tolist()
                return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            before = digest()
            model = retrieve._get_reranker(settings['RERANKER_MODEL'])
            report.update(snapshot_revision=snapshot.revision, snapshot_sha256=before,
                          score_contract=score_contract(model, settings['RERANKER_MODEL']),
                          settings=settings, generation_evaluated=args.generation)
            arms = [('baseline', {}), ('threshold_sensitivity_0.05', {'RERANKER_MIN_SCORE': '.05'}),
                    ('task_aware_caps', {'PAPER_CAP_POLICY': 'task-aware'})]
            if args.calibration:
                arms.append(('calibrated_threshold', {'RELEVANCE_CALIBRATION': args.calibration}))
            else:
                report['configurations']['calibrated_threshold'] = {'status': 'not_run', 'reason': 'No manually labeled calibration and held-out relevance examples supplied'}
            if args.generation:
                arms.append(('recovery_only', {'EVIDENCE_RECOVERY': 'true'}))
            else:
                report['configurations']['recovery_only'] = {'status': 'not_run', 'reason': 'Generation backend required; deterministic recovery tests run separately'}
            for name, overrides in arms:
                os.environ.update(settings | overrides)
                # Warm local model and corpus cache; use the same frozen snapshot.
                retrieve.retrieve('What is retrieval?', debug={})
                print('Evaluating', name, flush=True)
                measured = evaluate_dataset(args.dataset, retrieval_mode='hybrid-rerank',
                                            include_generation=args.generation, include_evidence_text=True)
                report['configurations'][name] = measured
                Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(measured['metrics']), flush=True)
            assert before == digest()
            assert snapshot.revision == retrieve._get_committed_snapshot(collection).revision
            report['snapshot_unchanged'] = True
            report['selected_settings'] = {'paper_cap_policy': 'baseline', 'reranker_min_score': .01,
                                            'score_transform': 'model-default', 'evidence_recovery': False,
                                            'reason': 'Compatibility baseline retained; no held-out manual calibration results'}
            Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
            write_summary(report, Path(args.output).with_suffix('.md'))
            # Exact passage annotation packet. Unlabeled until a human reviews it.
            examples = []
            baseline = report['configurations']['baseline']
            for i, case in enumerate(baseline['cases']):
                for passage in case['retrieved']:
                    examples.append({'query': case['question'], 'passage': passage['text'],
                                     'source': passage['source'], 'page': passage['page'],
                                     'split': 'held_out' if i % 3 == 0 else 'calibration',
                                     'kind': None, 'relevant': None, 'annotator': None})
            Path('evaluations/relevance-labels-to-review.json').write_text(json.dumps({'status': 'unlabeled', 'snapshot_revision': snapshot.revision, 'examples': examples}, indent=2) + '\n')
    finally:
        ingest.DB_DIR = original_db
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def write_summary(report, path='evaluations/evidence-comparison.md'):
    """Readable tradeoffs; never relabel locator proxies as passage relevance."""
    contract = report['score_contract']
    arms = [a for a in report['configurations'].values() if 'cases' in a]
    case_count = len(arms[0]['cases'])
    answerable = sum(c['expected'] == 'answered' for c in arms[0]['cases'])
    lines = [
        '# Evidence selection evaluation', '',
        'Measured with cached local embedding/reranker models; generation uses the configured backend.', '',
        f"Snapshot revision: `{report['snapshot_revision']}`. Document/text/metadata/vector digest: `{report['snapshot_sha256']}`.",
        f"Snapshot unchanged: **{report.get('snapshot_unchanged', False)}**. No PDFs, embeddings, or user index records were modified.", '',
        f"Reranker: `{contract['model']}`, revision `{contract['revision']}`, sentence-transformers `{contract.get('wrapper_version', 'unknown')}`.",
        f"Transformation: `{contract['transformation']}`; effective activation: `{contract['effective_activation']}`.",
        f"Score scale follows the recorded activation ({contract.get('score_scale', 'Sigmoid: [0,1]')}); these scores are not confidence probabilities.", '',
        f'{case_count} golden queries contain {answerable} answerable cases and {case_count-answerable} negative cases. Each arm changes one setting on the same index copy.',
        'Source/page labels support locator proxies only. True passage relevance precision/recall and blind human factual answer accuracy are **unavailable**, not zero.', '',
        '| Arm | Threshold | Locator precision | Locator recall | Verifier-supported answers | False abstentions | False answers | Retrieval errors | Validation failures |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    measured = []
    for name, arm in report['configurations'].items():
        if 'metrics' not in arm:
            lines += [f"| {name} | — | — | — | — | — | — | — | — |"]
            continue
        measured.append((name, arm))
        m = arm['metrics']
        contracts = arm['effective_settings'].get('score_contracts', [])
        threshold = contracts[0].get('effective_threshold', arm['effective_settings']['retrieval']['reranker_min_score']) if contracts else arm['effective_settings']['retrieval']['reranker_min_score']
        supported = sum(c.get('supported_answer', False) for c in arm['cases'] if c['expected'] == 'answered')
        failures = sum(c.get('actual') == 'validation_failed' for c in arm['cases'])
        support_display = f'{supported}/{answerable}' if arm.get('generation_included') else '—'
        failure_display = failures if arm.get('generation_included') else '—'
        precision = f"{m['anchor_locator_precision']:.1%}" if m['anchor_locator_precision'] is not None else '—'
        recall = f"{m['anchor_locator_recall']:.1%}" if m['anchor_locator_recall'] is not None else '—'
        lines.append(f"| {name} | {threshold:g} | {precision} | {recall} | {support_display} | {m.get('false_abstentions', '—')} | {m.get('false_answers', '—')} | {m.get('retrieval_failures', 0)} | {failure_display} |")
    calibration_note = ('Calibrated threshold: not run; no manually labeled query/passage calibration or held-out examples were supplied.'
                        if report['configurations']['calibrated_threshold'].get('status') == 'not_run' else
                        'Calibrated threshold results use the separately supplied, model-compatible manual calibration artifact.')
    lines += ['', calibration_note + ' The 0.05 arm is exploratory threshold sensitivity, not calibration.', '',
              '| Arm | Retrieval ms/query | Generation + validation + recovery ms/query | Embedding calls | Reranker calls | LLM request attempts | Recovery success |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, arm in measured:
        m = arm['metrics']
        successes = sum(c.get('recovery', {}).get('recovery_success', False) for c in arm['cases'])
        attempts = m.get('recovery_attempts', 0)
        lines.append(f"| {name} | {m['mean_retrieval_latency_ms']:.0f} | {m.get('mean_generation_latency_ms', 0):.0f} | {m['query_embedding_calls']} | {m['reranker_calls']} | {m.get('model_calls', 0)} | {successes}/{attempts} |")
    papers = sorted({p for _, arm in measured for p in arm['metrics']['per_paper_coverage']})
    label = lambda p: 'Bias survey' if p.startswith('Biases') else 'RAG survey' if p.startswith('Trustworthiness') else p
    lines += ['', 'Local embedding/reranker counts include recovery retrieval. LLM attempts include generation, retries, verification, and failed attempts; each question has a shared six-call LLM budget in this run. Recovery is limited to one additional retrieval round.',
              'Recovery success means new evidence is actually cited in a semantically supported answer. It does not establish that every original gap was resolved; that needs human review. Supported partial answers are reported separately in case debug metadata.', '',
              '| Arm | ' + ' | '.join(label(p) + ' recall' for p in papers) + ' |',
              '| --- | ' + ' | '.join('---:' for p in papers) + ' |']
    for name, arm in measured:
        values = arm['metrics']['per_paper_coverage']
        recalls = [f"{values[p]['recall']:.1%}" if p in values and values[p]['recall'] is not None else '—' for p in papers]
        lines.append('| ' + name + ' | ' + ' | '.join(recalls) + ' |')
    lines += ['', 'Paper recall credits only the matching retrieved paper; another paper retrieved for the same comparison does not earn credit. Dimension coverage is not measured on this focused-query set because it has no dimension labels. Fair allocations, section coverage, opposing positions with conditions, and missing dimensions are covered by deterministic tests, not claimed as corpus benchmark results.', '',
              'Selected settings remain the compatibility baseline: `RERANKER_MIN_SCORE=0.01`, `RERANKER_SCORE_TRANSFORM=model-default`, `PAPER_CAP_POLICY=baseline`, `EVIDENCE_RECOVERY=false`.',
              'The compatibility baseline had a whole-anchor context-budget failure for the bias-definition case in this run; the task-aware arm admitted whole anchors within budget and avoided that failure. Recall includes failed cases, and the report preserves their error messages.\nThere is no evidence supporting a calibrated new default. Higher page recall can bring more irrelevant passages or unsupported answers. Generation sampling and machine load vary; these single-pass latency and answer-support results have no confidence intervals. Thresholding also affects MMR selection, so locator recall need not change monotonically.', '',
              'The semantic verifier uses the generation model and can err. Its approval is distinct from factual correctness and answer completeness. JSON reports preserve generated answers, answer digests, verdicts, errors, retrieval provenance, score contracts, rejected candidates, allocations, and recovery queries for human review.', '',
              'Remaining work: manually review relevance examples including difficult topical negatives and out-of-corpus questions in both query-disjoint splits; evaluate summaries/comparisons/conflicts with section, dimension, position, and study-condition labels; run blind answer review and repeated held-out comparisons before promoting settings.', '',
              'No index migration or rebuild is needed. Custom rerankers and activation overrides require an explicit model-specific threshold or compatible calibration artifact. Overlapping recovery blocks are conservatively skipped to preserve original evidence IDs; new expanded blocks are conservatively charged to the remaining expansion allowance.', '',
              f"Test validation recorded separately: {report.get('test_validation', 'not run by the benchmark command; run python -m pytest -q')}.", '']
    Path(path).write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
