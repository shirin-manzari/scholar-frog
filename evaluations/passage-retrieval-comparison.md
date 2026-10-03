# Passage retrieval comparison — 2026-10-04

Compared the pre-change character chunker and neighbor-slot selection with the
new token chunker and paragraph expansion, using the two existing PDFs (63
extracted pages total) and all five questions in `golden-v1.json`. Both indexes
were built in isolated temporary directories. The active user index and PDF
files were left untouched. Extraction was shared between the runs, and cached
models were used offline. Each pipeline had a warm-up evaluation followed by
one measured evaluation. Latency includes retrieval, reranking, and expansion;
it excludes generation. These five queries are a small check, not a robust
latency benchmark or a broader quality evaluation.

The embedding model remained `BAAI/bge-small-en-v1.5`; the CPU reranker remained
`BAAI/bge-reranker-base`. Both used hybrid-rerank, top-k 5, the existing relevance
threshold (0.01), MMR (0.75), and paper cap (2). Dense and BM25 each
used 50 candidates; the reranker considered 40. Raw results are recorded in
`passage-retrieval-comparison.json`. The old index used 800 characters/150 character
overlap; the new index used 300 tokens/about 50 token overlap. New context used
one neighboring paragraph, a 4,000-token evidence budget, an 8,192-token prompt
window, and a 1,024-token answer reserve.

| Measure | Old | New |
| --- | ---: | ---: |
| Supported questions with expected page among selected anchors | 4/4 | 4/4 |
| Supported questions with expected page in final evidence | 4/4 | 4/4 |
| Bias survey page coverage | 2/2 | 2/2 |
| RAG survey page coverage | 2/2 | 2/2 |
| Out-of-corpus question: returned evidence blocks | 0 | 0 |
| Mean warm retrieval latency, all five questions | 2,964.71 ms | 3,476.34 ms |
| Mean formatted context, four supported questions | 423.75 tokens | 1,077 tokens |
| Largest formatted context | 726 tokens | 1,885 tokens |
| Stored search passages | 556 | 331 |
| Largest search passage, including special tokens | 484 tokens | 302 tokens |
| Repeated stored page-context payload | 0 | 1,801,624 bytes |

Context counts use the unchanged embedding tokenizer and include evidence IDs,
source/page labels, and separators. Answer-reserve and question/instruction
counts are additional. Original anchor scores and locations remain attached to
merged evidence. Page coverage measures source/page presence, not semantic
support, and final evidence coverage is reported separately from anchor coverage.

| Question | Old context tokens | New context tokens |
| --- | ---: | ---: |
| Data selection bias definition | 726 | 1,885 |
| Wikipedia editor demographics | 387 | 644 |
| TRC Bench scope and number of LLMs | 281 | 744 |
| Complete RAG system stages | 301 | 1,035 |
| Medieval astronomy | 0 | 0 |

Spot-checking the retrieved text confirmed that both pipelines supplied the
bias definition, the reported 87% figure, TRC Bench's six dimensions and 19
LLMs, and the three RAG stages. This is evidence availability, not a generated
answer assessment. Ollama was unavailable, so no generated-answer quality,
semantic-support, or answer-latency comparison was possible. No better-answer
claim follows from the retrieval results.

The new format supplies more surrounding context at about 17% higher warm
retrieval latency in this run. Although its largest search passage is smaller,
a 300-token target produces fewer passages than the old 800-character target
on these PDFs; token and character targets are not equivalent. Page text is
repeated in chunk metadata to inherit the existing commit/recovery protocol,
which increases storage. Sentence segmentation and Markdown section detection
are heuristic; overlap can be below its target when complete sentences do not
fit. Backend tokenization and framing may differ from the local budget estimate.

Validation: baseline suite **144 passed**; completed suite **162 passed**, including
sentence/paragraph preservation, capacity and oversized-sentence checks, complete
text coverage with overlap, section/page/document/version boundaries, deduplication,
budget limits, committed-only expansion, exact citation mapping, incompatible-index
rejection, interrupted-write recovery with stored context, and the existing staged
rebuild failure-retention test.

Before migrating, update any configured chunking strategy to `sentence-token`
and replace old character-size overrides with token settings (defaults 300/50).
Then explicitly rebuild:

```sh
python ask.py index rebuild --papers papers
```
