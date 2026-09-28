# ScholarQ

Ask questions about local research PDFs and get answers with paper and page
references. Requires Python 3.10+.

## Quick start

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen3:8b
```

Put text-based PDFs in `papers/`, start Ollama, then run:

```sh
python ask.py "What methods do these papers use?"
```

## Local web UI

Run `python app.py`, then open <http://127.0.0.1:8765>. Add PDFs in the
browser or place them in `papers/`, click **Sync library**, and ask a question.
The UI shows validated evidence references and lets you open the cited PDF
page. Messages remain visible in the current tab, but each question is
answered independently; refreshing the page clears the chat view.
The server listens only on your computer; stop it with Ctrl+C. The first sync
may download local embedding models and take a few minutes.

PDFs, embeddings, and the index remain local. Scanned PDFs are unsupported.
OpenAI and Anthropic are optional answer-generation backends; configure their
API keys in `.env` and install the matching package.

## Common commands

```sh
python ask.py --ingest-only
python ask.py --top-k 8 "Compare the methods"
python ask.py --debug-citations "Question"
python ask.py sync --dry-run
python ask.py index info
python ask.py index check
python ask.py index rebuild
```

Search defaults to semantic retrieval, BM25, and reranking. Changed PDFs are
synced automatically. `--compare` compares retrieval modes.

## Answers and index

Substantive answers require valid `[E1]`-style references. ScholarQ checks that
references exist, but does not verify factual support. If evidence is
insufficient, it returns a fixed abstention without references. Malformed or
uncited answers receive up to one repair attempt by default; unresolved
failures are hidden.

Configure indexing in `scholarq.toml` (copy `scholarq.toml.example`). Precedence
is defaults, TOML, environment variables, then supported CLI options. Changes
to embedding, chunking, or extraction settings require a rebuild. Rebuild
verifies a replacement before activation and retains the prior index. Legacy
indexes must be rebuilt explicitly.
