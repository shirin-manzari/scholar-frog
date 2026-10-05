# Scholar Frog

Ask a question about your PDFs. Scholar Frog finds relevant passages and gives an answer with links to the paper and page. If the evidence is thin, it says so.

PDF processing and search run on your machine. Ollama handles answers locally by default.

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

OpenAI and Anthropic are optional. Set `LLM_BACKEND` and its API key in `.env`, then install `openai` or `anthropic`. Those backends receive your question and retrieved passages.

## Limits

Scanned PDFs need OCR and are skipped. Citations are checked for valid passage IDs, and semantic verification is enabled by default, but you should still check important claims against the papers. The first run may download the local embedding and reranking models.
