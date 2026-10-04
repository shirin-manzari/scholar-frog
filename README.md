# Scholar Frog

Ask questions about your research papers and get answers with citations you can
check, right down to the page. When the papers don’t support an answer, Frog
says so.

Your PDFs and search stay on your machine. Answers use Ollama locally by default.

## Get started

You’ll need Python 3.10+, [Ollama](https://ollama.com), and PDFs with selectable
text. Scanned papers aren’t supported yet.

With Ollama running:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env
ollama pull qwen3:8b
python app.py
```

Open <http://127.0.0.1:8765>, add your papers, and sync the library. Then ask a
question. Each answer links back to the evidence used to write it.

Prefer the terminal? Put PDFs in `papers/` and run:

```sh
python ask.py sync
python ask.py "What methods do these papers use?"
```

## Make it yours

Model and search settings live in [.env.example](.env.example). Index settings
are in [scholar-frog.toml.example](scholar-frog.toml.example).

OpenAI and Anthropic are optional. Choose a backend in `.env`, add its API key,
and install its SDK with `pip install openai` or `pip install anthropic`.
These backends send your question and retrieved passages to the provider.

Frog includes nearby paragraphs to give evidence context. If a question hits the
context limit, try a shorter question or fewer results with `--top-k 3`.
Only increase the context window within your model’s supported limits.

## Trying the search experiments

The default BGE model uses a query instruction automatically
(`QUERY_INSTRUCTION=auto`). This affects search only and needs no index rebuild.

Query expansion is experimental and **off by default**. Set
`QUERY_EXPANSION=true` to try acronym definitions found in your indexed papers.
Frog keeps your original question and adds at most two search variants.
Set `QUERY_MAX_VARIANTS=0` to search only your original wording.

To see what search is doing:

```sh
python ask.py "your question" --no-ingest --debug-retrieval
```

Our small [evaluation](evaluations/query-processing-comparison.md) found no
recall improvement from expansion. Keep it off unless it helps on your papers;
retrieval scores alone don’t tell us whether answers are better.

If Frog asks you to rebuild an older index:

```sh
python ask.py index rebuild --papers papers
```

Your PDFs stay untouched, and a failed rebuild keeps the previous index.
