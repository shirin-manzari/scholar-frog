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

To inspect or run library synchronization directly:

```sh
python ask.py sync --dry-run
python ask.py sync
python ask.py sync --force
```

ScholarQ scans the paper folder, including subfolders, and tracks PDFs in
`chroma_db/documents.json`. SHA-256 content hashes identify documents;
identical copies share one set of chunks, with the alphabetically first
relative path used for citations. Renames update citation paths without
re-embedding. Modified PDFs are indexed and verified before old chunks are
removed. Deleted files are removed from the index, not from disk. A deletion
batch above `SYNC_DELETE_THRESHOLD` (default: 50% of indexed documents)
requires `--force`.

The manifest is a recovery aid; ScholarQ checks Chroma records before
reusing an entry. If interrupted during an index write, the pending manifest
marker lets the next sync safely retry. Failed updates retain the previous
indexed version where available. Dense and BM25 retrieval read the same
Chroma collection; changes invalidate the in-memory BM25 cache. The dry run
shows planned operations and does not write the manifest or collection.

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
# Versioned index configuration

ScholarQ records the indexing configuration used to create the active ChromaDB collection in `chroma_db/index_metadata.json`. The default settings preserve the original index behavior:

An index marked `ready` has version metadata and a persisted manifest describing its committed contents. A synchronization can still report failed PDFs while the successfully committed documents remain usable; failures remain retryable on the next sync.

Retrieval filters every Chroma result against the active document manifest. In-process sync and retrieval use a shared lock for a consistent snapshot; concurrent writers in separate ScholarQ processes are not coordinated and should be avoided.

| Setting | Default |
| --- | --- |
| `embedding_model` | `BAAI/bge-small-en-v1.5` |
| `embedding_revision` | unset (model repository default; set a commit to pin it) |
| `embedding_dimension` | `384` |
| `normalize_embeddings` | `true` |
| `chunk_size` | `800` |
| `chunk_overlap` | `150` |
| `chunking_strategy` | `markdown-heading-paragraph` |

Copy `scholarq.toml.example` to `scholarq.toml` to configure the index. Implementation versions for chunking and PDF text extraction are maintained by ScholarQ and are included in the fingerprint. Settings are read in this order: built-in defaults, `[index]` values in `scholarq.toml`, `SCHOLARQ_*` environment variables, then explicit command-line overrides (where offered). The fingerprint includes only settings that affect indexed text or vectors; retrieval mode, BM25, reranking and generation settings do not trigger reindexing.

Before retrieval or synchronization, ScholarQ compares the active settings with the stored fingerprint and schema version. A mismatch stops use of the index with an explanation; ordinary questions never trigger an automatic rebuild. Inspect it with:

```bash
python ask.py index info
python ask.py index check
```

Rebuild into a separate staging collection, verify its chunks and vectors, and activate it only after success:

```bash
python ask.py index rebuild --papers papers
```

The prior collection is retained. An interrupted or failed rebuild leaves it active; an unused staging collection may remain and is ignored. PDF files are never modified. Existing collections with no version metadata are treated as legacy and are not silently adopted because their exact model revision and extraction behavior cannot be established. Rebuild them explicitly; the old collection is retained after activation. Document synchronization stores the configuration fingerprint per document and avoids re-embedding unchanged PDFs when both their content and index configuration match. BM25 is rebuilt from the active Chroma collection and its cache is invalidated after activation.
