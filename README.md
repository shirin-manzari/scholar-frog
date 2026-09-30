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
