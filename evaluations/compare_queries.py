"""Four-way offline query experiment on one isolated committed index snapshot.

Run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m evaluations.compare_queries
Baseline is the pre-change retrieval source at QUERY_BASELINE_REF. PDFs/index
are not reprocessed. Labels in query-golden-v1.json must be reviewed manually.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import asdict
import tempfile

from src import ingest, retrieve, sync
from src.evaluate import evaluate_dataset
from src.context_budget import generation_token_counter


def main():
    baseline_ref = os.getenv('QUERY_BASELINE_REF', 'a0b2324cd51983642e4b4a0aba9ddac8379cca45')
    # Explicit effective settings, independent of a user's .env.
    settings = {'DENSE_CANDIDATES': '50', 'BM25_CANDIDATES': '50', 'RRF_K': '60',
                'RERANK_CANDIDATES': '40', 'FINAL_RESULTS': '5',
                'RERANKER_MODEL': 'BAAI/bge-reranker-base', 'RERANKER_MIN_SCORE': '0.01',
                'MAX_CHUNKS_PER_PAPER': '2', 'MMR_LAMBDA': '0.75',
                'CONTEXT_PARAGRAPHS': '1', 'CONTEXT_EXPANSION_TOKENS': '1500',
                'CONTEXT_SAME_SECTION': 'true', 'CONTEXT_BUDGET_TOKENS': '4000',
                'CONTEXT_WINDOW_TOKENS': '8192', 'CONTEXT_ANSWER_RESERVE': '1024',
                'CONTEXT_FRAMING_RESERVE': '32', 'CONTEXT_RETRY_RESERVE': '256',
                'QUERY_MAX_VARIANTS': '2', 'LLM_BACKEND': 'ollama'}
    previous = {key: os.environ.get(key) for key in settings | {'QUERY_INSTRUCTION': '', 'QUERY_EXPANSION': ''}}
    original_db = ingest.DB_DIR
    dataset = Path('evaluations/query-golden-v1.json')
    try:
        os.environ.update(settings)
        with tempfile.TemporaryDirectory(prefix='scholar-frog-queries-') as root:
            root = Path(root)
            with sync.INDEX_LOCK:
                shutil.copytree(original_db, root / 'chroma_db')
            ingest.DB_DIR = str(root / 'chroma_db')
            source = subprocess.check_output(['git', 'show', f'{baseline_ref}:src/retrieve.py'])
            baseline_path = root / 'baseline.py'
            baseline_path.write_bytes(source)
            spec = importlib.util.spec_from_file_location('query_baseline', baseline_path)
            baseline = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = baseline
            spec.loader.exec_module(baseline)
            baseline._get_reranker = retrieve._get_reranker
            collection = ingest.get_collection()
            snapshot = retrieve._get_committed_snapshot(collection)
            _, corpus, _ = retrieve._get_bm25_index(collection, snapshot)
            before = collection.get(ids=sorted(snapshot.chunk_ids), include=['embeddings'])
            vectors_hash = hashlib.sha256(json.dumps(before['embeddings'].tolist(), sort_keys=True).encode()).hexdigest()
            report = {'snapshot_revision': snapshot.revision, 'stored_passages': len(snapshot.chunk_ids),
                      'baseline_ref': baseline_ref, 'baseline_source_sha256': hashlib.sha256(source).hexdigest(),
                      'document_embeddings_sha256': vectors_hash, 'settings': settings,
                      'token_counter': generation_token_counter().method,
                      'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                      'generation_evaluated': False, 'configurations': {}}
            for label, module, instruction, expansion in [
                ('baseline', baseline, 'off', 'false'),
                ('instruction_only', retrieve, 'auto', 'false'),
                ('expansion_only', retrieve, 'off', 'true'),
                ('both', retrieve, 'auto', 'true'),
            ]:
                os.environ.update(QUERY_INSTRUCTION=instruction, QUERY_EXPANSION=expansion)
                print('Warming', label, flush=True)
                evaluate_dataset(dataset, top_k=5, retrieval_mode='hybrid-rerank', retrieve_fn=module.retrieve)
                measured = evaluate_dataset(dataset, top_k=5, retrieval_mode='hybrid-rerank', retrieve_fn=module.retrieve, include_evidence_text=True)
                # Retain supplied text, anchor provenance and scores for manual relevance review.
                if label == 'baseline':
                    measured['effective_settings']['retrieval'] = {**asdict(baseline.RetrievalConfig.from_env()), 'mode': 'hybrid-rerank'}
                measured['effective_settings']['baseline_pipeline'] = label == 'baseline'
                measured['effective_settings']['query_variants_policy'] = ('replacement typo correction' if label == 'baseline' else 'original plus bounded variants')
                measured['snapshot_revision'] = snapshot.revision
                report['configurations'][label] = measured
                print(label, json.dumps(measured['metrics']), flush=True)
            after = collection.get(ids=sorted(snapshot.chunk_ids), include=['embeddings'])
            assert vectors_hash == hashlib.sha256(json.dumps(after['embeddings'].tolist(), sort_keys=True).encode()).hexdigest()
            assert retrieve._get_committed_snapshot(collection).revision == snapshot.revision
            report['document_embeddings_unchanged'] = True
            report['terminology'] = retrieve._get_terminology(collection, snapshot, corpus)
            Path('evaluations/query-processing-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        ingest.DB_DIR = original_db
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == '__main__':
    main()
