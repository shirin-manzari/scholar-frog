# Scholar Frog

Ask questions about your papers and get answers with page-linked evidence.
If the papers don’t support an answer, Frog says so.

PDF processing and search stay on your machine. Answers run locally through
Ollama by default; OpenAI and Anthropic are optional.

## Get started

You’ll need Python 3.10+, Ollama, and PDFs with selectable text. Scanned PDFs
aren’t supported yet.

Start Ollama, then run:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env
ollama pull qwen3:8b
python app.py
```

Open <http://127.0.0.1:8765>, add your PDFs, and sync the library. You can also
put papers in `papers/` and use the terminal:

```sh
python ask.py sync
python ask.py "What methods do these papers use?"
```

## Settings and updates

See [.env.example](.env.example) for model and context settings, and
[scholar-frog.toml.example](scholar-frog.toml.example) for index settings.
Cloud backends need their API key and SDK (`pip install openai` or
`pip install anthropic`) and send the question and evidence to that provider.

Search passages target 300 tokens with about 50 tokens of overlap, keeping
sentences intact where possible. Frog adds nearby paragraphs for context and
keeps each page’s evidence separate. Context limits use the local tokenizer
and reserve room for the question, instructions, and answer; your answer
model’s token counts may differ.

Upgrading an older index? Change any configured `chunking_strategy` to
`sentence-token` and replace old character-size settings with token settings
(defaults: 300/50). Then rebuild:

```sh
python ask.py index rebuild --papers papers
```

A failed rebuild keeps the previous index. Your PDFs are never changed.

## Check retrieval

```sh
python ask.py evaluate --output evaluation-results/latest.json
```

Add `--generate` to check answers too; it uses your configured model.
See the [retrieval comparison](evaluations/passage-retrieval-comparison.md) for
results and limitations. Finding the right page alone doesn’t prove an answer
is correct.
