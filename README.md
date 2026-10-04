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

## Evidence selection and calibration

The compatibility baseline stays at `RERANKER_MIN_SCORE=0.01`,
`RERANKER_SCORE_TRANSFORM=model-default`, `PAPER_CAP_POLICY=baseline`, and
`EVIDENCE_RECOVERY=false`. This threshold is **not calibrated**. The installed
CrossEncoder 6.1.0 uses Sigmoid for the cached one-output BGE reranker; neither
Sigmoid scores nor raw logits are confidence probabilities. Debug output and
reports record activation, model revision, threshold, rejected candidates, and
allocations. A calibration artifact for a different model, revision, or
transformation is rejected. Pin `RERANKER_REVISION` for reproducible scores. Changing the reranker model or
activation requires an explicit threshold or matching calibration artifact; the
BGE compatibility threshold is never silently inherited in that case.

To calibrate offline, manually label query/passage pairs, including relevant
passages, difficult topical negatives, and out-of-corpus questions in **each**
split. Keep whole queries separate between `calibration` and `held_out`. Start
from `evaluations/relevance-labels-to-review.json`; its null labels are deliberately
unusable until reviewed. Add missing negative examples rather than treating every
unmatched page as irrelevant. Each example needs this structure:

```json
{"query":"What methods were used?","passage":"Exact passage text",
 "split":"calibration","kind":"relevant","relevant":true,"annotator":"reviewer"}
```

Wrap examples in `{"examples":[...]}`. Other kinds are `topical_negative` and
`out_of_corpus`, both with `relevant:false`. Then run:

```sh
python -m evaluations.calibrate_relevance labels.json calibration.json \
  --thresholds .001 .01 .05 .1 --select .05
```

The command uses cached local models only, includes the current threshold as a
baseline, and reports precision, recall, false rejections, and false acceptances
for every candidate in both splits. `--select` is an explicit operator choice:
consider both acceptance and rejection costs on calibration data, then assess
held-out results without retuning on that split. No threshold is auto-selected.
Set `RELEVANCE_CALIBRATION=calibration.json` only after reviewing the results.
Changing relevance policies requires no PDF edits, embedding changes, or index
rebuild. Artifacts must match the active reranker revision and score transformation.

Set `PAPER_CAP_POLICY=task-aware` to try whole-anchor selection within the existing
context and evidence budgets. Focused single-paper questions may use the full
anchor budget. Summaries select available sections before repeated sections;
comparisons rotate across explicitly selected papers and requested dimensions.
Conflict plans can specify literal positions including study-condition terms.
Missing relevant evidence is recorded before allocation is redistributed. Quotas
never lower the relevance threshold. Structural summary candidates now pass through
reranking within the same candidate allowance. This policy requires hybrid-rerank.

Programmatic callers can pass an optional plan without changing the chunk contract:

```python
from src.evidence_policy import RetrievalPlan
from src.retrieve import retrieve
plan = RetrievalPlan(task="comparison", papers=("A.pdf", "B.pdf"),
                     dimensions=("accuracy", "latency"))
chunks = retrieve("Compare A and B on accuracy and latency", plan=plan, debug={})
```

Plans also accept `subqueries` using the existing shared query-variant allowance,
and `positions` for conflict analysis. Automatic task inference is conservative;
explicit plans are preferable for complex dimensions or study conditions. Dimension
matching is literal, not a semantic classifier. Context expansion never spends
paper quotas. Whole passages that cannot fit are skipped and recorded, never cut.

Set `EVIDENCE_RECOVERY=true` to allow at most one additional retrieval round after
valid semantic-verifier verdicts identify unsupported claims, or a structured
insufficient-evidence response identifies an exact question excerpt as a missing
fact. Malformed JSON, invalid citation IDs, and verifier outages do not initiate
search. Recovery retains the original question verbatim and marks any unsupported
claim as a proposition to investigate. Recorded explicit paper scope and committed
revision are preserved. Duplicate or overlapping same-version blocks are skipped;
new blocks append evidence IDs while sharing anchor, paper, context, and expansion
budgets. A changed snapshot excludes recovery evidence. After regeneration, both
citation and semantic validation run again; a supported partial answer is allowed,
and another unsupported response becomes an abstention. `GENERATION_TOTAL_CALLS`
bounds all generation/verifier requests across ordinary retries and recovery.
Generation result `debug` includes recovery reasons, queries, counts, and usage.
Caller-supplied recovery callbacks must return committed evidence in the original
scope; normal callers use the recorded scope automatically.

Run comparable experiments on an isolated copy of the existing index:

```sh
python -m evaluations.compare_evidence
python -m evaluations.compare_evidence --calibration calibration.json --generation
python -m pytest -q
```

Each arm measures a single policy change. Reports separate true passage relevance
metrics (requiring manual labels) from coarse page-locator proxies. Paper recall
credits only retrieved papers. Answered status is format agreement, not factual
accuracy; semantic support is reported separately. Blind answer reviews must match
the SHA-256 of the generated answer via `answer_review.answer_sha256` and
`answer_review.correct`. See [evaluation results](evaluations/evidence-comparison.md)
for measured tradeoffs and current limitations. No index migration is required;
custom rerankers or activation overrides need an explicit threshold setting
or compatible calibration artifact.
