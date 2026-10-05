# Scholar Frog

This is a self-education project for learning how a local RAG pipeline works.

Ask a question about your PDFs. Scholar Frog finds relevant passages and gives an answer with links to the paper and page. If the evidence is thin, it says so.

PDF processing and search run on your machine. Ollama handles answers locally by default.

## How it works

1. Sync PDFs: extract text by page, split it into passages, and store local embeddings in Chroma.
2. Ask a question: search passages with vector and keyword retrieval, then optionally rerank them.
3. Build context: select relevant passages and nearby paragraphs within the answer budget.
4. Generate an answer: send that context to Ollama by default, then check citation IDs and claim support.

## Start

You need Python 3.10+, [Ollama](https://ollama.com), and PDFs with selectable text.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen3:8b
python app.py
```

Open <http://127.0.0.1:8765>. Add PDFs, sync the library, then ask a question.

## Use the terminal

Put PDFs in `papers/`, then run:

```sh
python ask.py sync
python ask.py "What methods do these papers use?"
```

Useful commands:

```sh
python ask.py sync --dry-run                 # Preview library changes
python ask.py "your question" --paper paper.pdf
python ask.py "your question" --debug-retrieval
python ask.py index info                     # Inspect the index
python ask.py index rebuild --papers papers  # Rebuild an incompatible index
python -m pytest -q                          # Run tests
```

Use `python ask.py --help` for all options. Run commands from the project folder.

## Settings

Copy [.env.example](.env.example) to `.env` to change the answer model, retrieval mode, or evidence limits. Index settings are in [scholar-frog.toml.example](scholar-frog.toml.example). Search experiments and calibration are documented in [ARCHITECTURE.md](ARCHITECTURE.md).

Ollama can run other local chat models: pull one and set `OLLAMA_MODEL` in `.env`. The model needs to follow Scholar Frog's JSON and citation instructions, so results may vary.

OpenAI and Anthropic are the supported API backends. Set `LLM_BACKEND` and its API key in `.env`, then install `openai` or `anthropic`. Those backends receive your question and retrieved passages. Other API providers and non-Ollama local servers are not integrated yet.
