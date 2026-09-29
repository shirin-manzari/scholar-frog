# Scholar Frog

Ask questions about local PDFs and get answers linked to paper excerpts. Scholar Frog
runs ingestion and retrieval locally; Ollama is the default answer model.

## Start

Requires Python 3.10+ and Ollama.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen3:8b
python app.py
```

Open <http://127.0.0.1:8765>, add text-based PDFs, sync the library, and ask a
question. Make sure Ollama is running. The first sync may download local models.

For the CLI, put PDFs in `papers/` and run:

```sh
python ask.py "What methods do these papers use?"
```

Answers show evidence references you can open in the source PDFs. If the excerpts
do not support an answer, Scholar Frog says so. Check cited passages before using
an answer in academic work. Scanned PDFs need OCR and are not supported.

By default, every generated answer receives a second semantic verification pass:
each cited claim is checked against the exact excerpts attached to it. Unsupported
claims trigger one bounded regeneration attempt and are never shown if verification
still fails. This uses the configured LLM backend and can be disabled with
`CITATION_SEMANTIC_VALIDATION=false` or CLI option `--no-semantic-validation`.

Optional index settings go in `scholar-frog.toml` (see
`scholar-frog.toml.example`) or `SCHOLAR_FROG_` environment variables. The old
`scholarq.toml` and `SCHOLARQ_` names still work. OpenAI and Anthropic are
optional backends configured in `.env` with their SDKs installed separately.
