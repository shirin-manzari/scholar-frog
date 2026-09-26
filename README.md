# ScholarQ

Ask questions across your research papers from the command line. ScholarQ
searches local PDFs and gives answers with references such as `[E1]` that map
to a paper and page.

## Get started

Use Python 3.10 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add text-based PDFs to `papers/`. Start Ollama and download the default model:

```sh
ollama pull qwen3:8b
python ask.py "What methods do these papers use?"
```

The first run downloads the local embedding and reranking models. ScholarQ keeps
PDFs, embeddings, and its database on your machine. Scanned PDFs are not
supported yet. OpenAI and Anthropic are also available; set the backend and API
key in `.env` and install that provider's Python package.

## Useful commands

```sh
python ask.py --ingest-only                 # index papers without asking
python ask.py --no-ingest "What did they find?"  # skip the ingestion check
python ask.py --top-k 8 "Compare the methods"    # retrieve more passages
python ask.py --retrieval dense "Question"       # choose dense search
python ask.py --debug-citations "Question"      # inspect citation checks
```

New or changed PDFs are picked up automatically. Use `--reingest` to rebuild
the index after changing how PDFs are processed. `--compare` prints results
from each retrieval mode.

## Citations

ScholarQ assigns `[E1]`, `[E2]`, and so on to the final retrieved passages for
each question. It checks that cited IDs exist in the answer context, then lists
the cited papers and available page numbers under **References**. Unknown,
malformed, repeated, or missing citations trigger one repair attempt by
default; if the answer still fails, ScholarQ marks the failure and does not
show references as verified.

This check confirms that a cited passage was provided to the model. It does
not confirm that the passage supports the claim. Read important passages in
the original PDF. ScholarQ can also warn about sentences that may lack
citations; those warnings are heuristic and can be imperfect.

Use `--citation-retries 0` to skip repair attempts,
`--no-citation-validation` to opt out of reference checks, or
`--no-citation-coverage` to turn off coverage warnings. `--debug-citations`
shows the evidence map, validation errors, retry count, and original response.
Defaults and retrieval settings are in `.env.example`.

## Search options

The default `hybrid-rerank` mode combines local semantic search and BM25
keyword search, then reranks the results. Use `--retrieval dense` or
`--retrieval hybrid` to choose another mode. `--top-k` changes how many
passages are retrieved. See `.env.example` for candidate counts and model
settings.
