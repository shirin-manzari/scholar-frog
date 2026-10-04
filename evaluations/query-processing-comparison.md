# Experimental query processing — 2026-10-04

The four configurations used the same isolated copy of the existing **331-passage,
two-paper committed index**, revision
`46a36a25a9dcb878f01a37616957cf9cfbe133bdbbbb608ef8f02b631051e8a8`.
Document embeddings were not rebuilt; their SHA-256 digest was checked before
and after the experiment. The baseline retrieval module was loaded read-only
from Git commit `a0b2324cd51983642e4b4a0aba9ddac8379cca45`. The revised
configurations include original-question retention and reranking, safe correction,
and bounded fusion. The dataset contains no intentional typos, so correction
changes are covered by tests rather than isolated by this experiment.

The ten questions comprise eight supported examples and two negative examples.
Labels were checked against stored text and the actual PDF pages: IPI and
malicious instructions on RAG survey p.13; RLHF on p.3; membership inference
attacks on p.23; TRC Bench's six dimensions and 19 LLMs on p.4; the three RAG
stages on p.6; data selection bias on bias-survey p.3; Wikipedia editors' 87%
male figure on p.6. A negated Wikipedia question requires rejecting its premise.
Medieval astronomy and a dose-specific 20 mg/L question are unsupported.
Query text, expected locators and review notes are in `query-golden-v1.json`.

Each configuration received a full warm-up pass and one measured pass. Models
were already cached, network access for model loading was disabled, and all
retrieval/reranking ran locally. Timing includes query encoding, BM25, fusion,
CPU reranking and context assembly; it excludes model loading and generation.
The order was baseline, instruction only, expansion only, both. This small,
single measured pass is not a statistically reliable latency benchmark.

All configurations used BAAI/bge-small-en-v1.5, normalized 384-dimensional
embeddings, hybrid-rerank, BAAI/bge-reranker-base on CPU, 50 total dense and
50 total BM25 candidates, RRF k=60, 40 rerank candidates, top-k 5, a 0.01
reranker threshold, MMR 0.75 and a two-anchor paper cap. Variants were limited
to two, with half the search budget reserved for the original and total variant
fusion weight capped at the original's weight per method. Context settings:
one same-section neighboring paragraph, 1,500 expansion tokens, 4,000 formatted
evidence tokens, 8,192 window tokens, 1,024 answer reserve, 32 framing reserve
and 256 retry reserve. No matching generation tokenizer was available, so context
accounting used the conservative UTF-8 byte estimator. Effective settings,
errors, queries, anchor ranks/scores and exact supplied passages are retained in
`query-processing-comparison.json`.

| Measure | Baseline | Instruction only | Expansion only | Both |
| --- | ---: | ---: | ---: | ---: |
| Expected page among delivered anchors | 75% (6/8) | 75% (6/8) | 75% (6/8) | 75% (6/8) |
| Expected page in supplied context | 75% (6/8) | 75% (6/8) | 75% (6/8) | 75% (6/8) |
| Anchor locator MRR | 0.75 | 0.75 | 0.75 | 0.75 |
| Context locator MRR | 0.5625 | 0.5625 | 0.5625 | 0.5625 |
| Unjudged-page block fraction, supported cases | 68.18% | 68.18% | 68.18% | 68.18% |
| Negative questions with evidence | 1/2 | 1/2 | 1/2 | 1/2 |
| Irrelevant blocks on negative questions | 3 | 2 | 2 | 2 |
| Mean warm retrieval latency (ms) | 3,638.98 | 3,506.81 | 3,500.77 | 3,471.34 |

The data-selection definition failed at context assembly in every configuration:
its mandatory anchors required 4,760 evidence tokens under the byte estimator,
exceeding the 4,000 evidence budget. That is a budget failure, not evidence of
semantic retrieval failure, and it counts as a failed delivered-evidence case.
No selected anchor was silently removed or truncated. Use a matching local
generation tokenizer, reduce top-k, or choose budgets appropriate to the actual
backend window before interpreting such failures as retrieval misses.

IPI failed to retrieve the expected p.13 evidence in the final context in every
configuration, despite a corpus-attested expansion to Indirect Prompt Injection.
This is a recall gap for future evaluation and tuning; the acronym feature does
not guarantee that a candidate survives original-question reranking/selection.
RLHF and MIA succeeded with and without expansion. The numerical and negated
Wikipedia cases supplied the same expected page across configurations. Retrieving
a passage that contradicts a premise does not prove the generator will correctly
reject that premise.

The medieval-astronomy query returned no evidence in every configuration. The
dose-specific query returned irrelevant evidence in every configuration: none
of its passages establishes a RAG accuracy result at 20 mg/L. The instruction
and expansion changed these passages without eliminating the failure. Empty
retrieval on one negative question is measurable; generation abstention on the
dose question is not. A local Ollama connection check failed, so answer quality,
semantic support, answer latency and model abstention correctness were not measured.

The unjudged-page fraction counts supplied evidence blocks outside the listed
expected source/page locators. Those blocks may be related background or useful
context; this metric is not a semantic irrelevant-evidence rate. In contrast,
the blocks on the two explicitly out-of-scope questions were inspected as
irrelevant to their requested facts. Ranking metrics use exact anchor rank
separately from the ordering of expanded page blocks; context can precede its
anchor page. Neither page recall nor these locator metrics establishes answer
quality.

Recommended defaults: **QUERY_INSTRUCTION=auto**, scoped only to the exact default
BGE model, **QUERY_EXPANSION=false**, and **QUERY_MAX_VARIANTS=2**. The instruction
follows the model's query guidance; this experiment supplies no evidence of a
recall benefit. Expansion remains opt-in: the current set shows no recall or
ranking gain. Alias coverage is intentionally limited to explicit initial-aligned
definitions; ambiguous, numeric and mixed-case abbreviations and arbitrary
synonyms are skipped. Unknown lowercase names can still resemble typos; use
`QUERY_MAX_VARIANTS=0` for exact-query experiments. Adding variants splits the
candidate budget and adds encoding/search work, which can hurt recall or latency
on a larger corpus. Evaluate more papers, genuinely difficult terminology,
ambiguous acronyms, paper-scoped questions, numerical constraints and negation
before considering broader coverage or changing defaults.

Validation: baseline suite **199 passed**; completed suite **237 passed**, including
instruction isolation/model selection, unchanged document encoding/index
compatibility, instruction-aware capacity boundaries, original retention and
reranking, alias ambiguity and scope, identifiers/names/numbers/units/negation,
shared plan/correction/alias limits, weighted deduplication, reserved original
candidates, committed-version and pending exclusion, cache revision changes,
merged provenance, optional debug compatibility, and effective evaluation settings.
CLI help and Python compilation checks passed. Existing generation-tokenizer
fallback warnings remain.
