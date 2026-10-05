# Scholar Frog contributor guide

Read this file and [ARCHITECTURE.md](ARCHITECTURE.md) before changing the retrieval or generation pipeline.

## Goal

Scholar Frog is a local-first research assistant for academic PDFs. It should answer from traceable page-level evidence and abstain when passages do not support an answer. Citation IDs are validated, and a second model call checks claim support by default. Neither check replaces human review.

## Current shape

- `app.py` and `web/` provide a local browser UI for upload, sync, ask, and source viewing.
- `ask.py` owns CLI parsing and terminal output. It also exposes `sync`, `index`, and `evaluate` commands.
- `src/ingest.py` extracts pages, cleans repeated margins, chunks passages, and sets up local embeddings and Chroma.
- `src/sync.py` plans and commits library changes through a manifest. Pending writes stay invisible to retrieval.
- `src/retrieve.py` runs dense, BM25, and optional cross-encoder search, then selects anchors and expands page context.
- `src/evidence_policy.py`, `src/relevance.py`, and `src/context_budget.py` handle task allocations, reranker scores, and generation budgets.
- `src/generate.py`, `src/citations.py`, and `src/recovery.py` handle answers, validation, and optional one-round evidence recovery.
- `src/evaluate.py` and `evaluations/` contain repeatable evaluations. `tests/` contains offline unit and integration tests.

## Contracts to preserve

- `ingest.ingest_folder(papers_dir)` syncs the library. `sync.sync_library()` owns manifest and index writes.
- `retrieve.retrieve(question, top_k=None, retrieval_mode=None, paper=None, *, debug=None, plan=None)` returns evidence dictionaries. Keep `text`, `source`, `title`, `page`, and `distance`; callers also use `id`, `section`, and `metadata`.
- `generate.generate_answer(question, chunks, ...)` returns a `GenerationResult`, including status, answer, evidence, and validation. It does not return a string.
- Keep the prompt restricted to supplied evidence. Every factual claim needs a valid evidence ID; the model must be able to abstain or identify unsupported parts.
- Keep ingestion, embeddings, and retrieval local. OpenAI and Anthropic are opt-in generation backends; their SDKs are optional.
- Keep source page numbers and character offsets attached to evidence. Never expose pending or uncommitted Chroma rows.
- Keep `ask.py` as orchestration. Put PDF and index logic in `ingest.py`/`sync.py`, search logic in `retrieve.py`, and prompt/provider logic in `generate.py`.

## Index changes

The schema-2 index fingerprint covers embedding, extraction, chunking, and context configuration. Changing those settings or algorithms requires an explicit `python ask.py index rebuild`. It stages and validates a replacement before activation. `python ask.py sync --dry-run` previews normal library changes. Deleted papers are removed from the index on sync; large deletion batches require `--force`.

## Working practices

- Run `.venv/bin/python -m pytest -q` after code changes. The suite uses fakes and does not need model downloads.
- Add a focused test for a behavior change, especially for citation mapping, retrieval scope, or index visibility.
- Do not commit `.env`, PDFs, or `chroma_db/`. PDFs in nested `papers/` folders are ignored too.
- Use short lowercase commit messages.
