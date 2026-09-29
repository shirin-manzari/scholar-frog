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
