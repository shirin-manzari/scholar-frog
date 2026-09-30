# Scholar Frog

Local-first Q&A for PDFs with page-linked evidence.

## Run

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen3:8b
python app.py
```

With Ollama running, open <http://127.0.0.1:8765>, add PDFs, and sync. CLI:

```sh
python ask.py "What methods do these papers use?"
```

Requires Python 3.10+, Ollama, and text-based PDFs. See `.env.example` for options.

## Web conversations

The web UI saves chats locally in `chroma_db/conversations.sqlite3` by default.
Use **+ New Chat** for a clean conversation; the current paper selection is
kept. Choose an older chat to restore its messages and paper scope. Deleting a
chat removes its saved turns but does not delete PDFs or the search index.

Follow-up questions are rewritten into standalone **search queries** using up
to `MAX_CONVERSATION_TURNS` recent turns (default 6). The resolver sees prior
user questions and cited paper IDs, not previous answer prose or PDF text. Each
factual turn runs fresh retrieval through the existing search pipeline; only
newly retrieved evidence can support the answer. The existing citation checks,
LLM support check, and abstention behavior still apply. The support check uses
the configured LLM, so it is a guardrail, not independent verification.

The local API supports `POST /api/conversations` with `{"document_ids":[]}`,
`GET /api/conversations`, `GET /api/conversations/{id}`,
`PATCH /api/conversations/{id}` with `{"document_ids":["paper.pdf"]}`,
and `DELETE /api/conversations/{id}`. Send a turn with `POST /api/ask` and
`{"conversation_id":"...","question":"...","document_ids":["paper.pdf"]}`.
An empty `document_ids` list searches the library. The response includes the
conversation ID, answer, citations, and resolved query. The old stateless
`/api/ask` request remains supported.

To run all tests:

```sh
python -m pytest -q
```

## Evaluate retrieval quality

The checked-in `evaluations/golden-v1.json` is a small, manually reviewed set
of questions for the two sample papers in `papers/`. It records expected source
pages, including one out-of-corpus question that should abstain. Run retrieval
metrics without calling an LLM:

```sh
python ask.py evaluate --output evaluation-results/latest.json
```

Add `--generate` to measure generated-answer outcome accuracy, abstention
correctness, and citation locator precision. It uses your configured LLM and
can therefore be slower or incur provider costs. Update the golden set only
after manually checking its source/page evidence; retain old versions so
quality changes remain comparable.
