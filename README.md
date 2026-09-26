# scholarq

scholarq is a command-line tool for asking questions across PDF papers. It
retrieves relevant passages from a local Chroma database and asks an LLM to
answer with citations in the form `[Paper Title, p.N]`. Check cited pages
against the PDFs before relying on an answer.

## Setup

Use Python 3.10 or newer. From the project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Place text-based PDFs in `papers/`. The first ingestion downloads the local
sentence-transformer model, which requires an internet connection. PDF text,
embeddings, and the vector database stay on this machine. Scanned PDFs are
skipped because OCR is not implemented.

The default generation backend is Ollama. Install and start Ollama separately,
then download the configured model (by default, `qwen3:8b`):

```sh
ollama pull qwen3:8b
```

Alternatively,
set `LLM_BACKEND=openai` or `LLM_BACKEND=anthropic` in `.env` and provide the
matching API key. Those backends require their optional SDKs (`openai` or
`anthropic`) to be installed separately.

## Use

```sh
python ask.py --ingest-only
python ask.py "What methods do the papers use?"
python ask.py --papers ./my-pdfs --top-k 8 "What are the main findings?"
python ask.py --no-ingest "How does paper A define the term?"
python ask.py --ingest-only --reingest
```

Ingestion is content-hash based: unchanged PDFs are skipped, and changed files
replace their prior chunks in the local database. `--no-ingest` skips the
folder check, so run ingestion after adding or changing papers. The database is
stored in `chroma_db/`. Use `--reingest` after changing extraction or chunking
logic to rebuild chunks for PDFs that were already indexed.

## Configuration

Settings are read from `.env`; see `.env.example` for backend names, model
settings, and service URL. Ollama is local. OpenAI and Anthropic send the
question and retrieved excerpts to the selected provider during generation.
`OLLAMA_TIMEOUT` controls the local generation timeout in seconds. Scholarq
disables Ollama thinking mode for supported models so it returns a direct
answer promptly.

## Design choices

PDFs are converted to Markdown before chunking. Chunks follow document
headings when available, with oversized sections split further while keeping
page references. Citations include section names when available, for example
`[Paper Title, Methods, p.4]`; unstructured excerpts use `[Paper Title, p.4]`.
