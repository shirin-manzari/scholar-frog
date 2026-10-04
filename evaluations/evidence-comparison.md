# Evidence selection evaluation

Measured with cached local embedding/reranker models; generation uses the configured backend.

Snapshot revision: `46a36a25a9dcb878f01a37616957cf9cfbe133bdbbbb608ef8f02b631051e8a8`. Document/text/metadata/vector digest: `9742977def90b9cebe2004d088f4fef2efd96857013323556cdd50c600d3f274`.
Snapshot unchanged: **True**. No PDFs, embeddings, or user index records were modified.

Reranker: `BAAI/bge-reranker-base`, revision `2cfc18c9415c912f9d8155881c133215df768a70`, sentence-transformers `6.1.0`.
Transformation: `model-default`; effective activation: `Sigmoid`.
Score scale follows the recorded activation ([0,1]); these scores are not confidence probabilities.

10 golden queries contain 8 answerable cases and 2 negative cases. Each arm changes one setting on the same index copy.
Source/page labels support locator proxies only. True passage relevance precision/recall and blind human factual answer accuracy are **unavailable**, not zero.

| Arm | Threshold | Locator precision | Locator recall | Verifier-supported answers | False abstentions | False answers | Retrieval errors | Validation failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibrated_threshold | — | — | — | — | — | — | — | — |
| baseline | 0.01 | 37.5% | 75.0% | 6/8 | 2 | 0 | 1 | 0 |
| threshold_sensitivity_0.05 | 0.05 | 41.2% | 87.5% | 7/8 | 1 | 0 | 0 | 0 |
| task_aware_caps | 0.01 | 34.5% | 100.0% | 8/8 | 0 | 0 | 0 | 0 |
| recovery_only | 0.01 | 37.5% | 75.0% | 6/8 | 2 | 0 | 1 | 0 |

Calibrated threshold: not run; no manually labeled query/passage calibration or held-out examples were supplied. The 0.05 arm is exploratory threshold sensitivity, not calibration.

| Arm | Retrieval ms/query | Generation + validation + recovery ms/query | Embedding calls | Reranker calls | LLM request attempts | Recovery success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 4070 | 20615 | 10 | 10 | 18 | 0/0 |
| threshold_sensitivity_0.05 | 5803 | 24152 | 10 | 10 | 16 | 0/0 |
| task_aware_caps | 4472 | 22719 | 10 | 10 | 19 | 0/0 |
| recovery_only | 3759 | 20641 | 12 | 12 | 21 | 0/2 |

Local embedding/reranker counts include recovery retrieval. LLM attempts include generation, retries, verification, and failed attempts; each question has a shared six-call LLM budget in this run. Recovery is limited to one additional retrieval round.
Recovery success means new evidence is actually cited in a semantically supported answer. It does not establish that every original gap was resolved; that needs human review. Supported partial answers are reported separately in case debug metadata.

| Arm | Bias survey recall | RAG survey recall |
| --- | ---: | ---: |
| baseline | 66.7% | 80.0% |
| threshold_sensitivity_0.05 | 100.0% | 80.0% |
| task_aware_caps | 100.0% | 100.0% |
| recovery_only | 66.7% | 80.0% |

Paper recall credits only the matching retrieved paper; another paper retrieved for the same comparison does not earn credit. Dimension coverage is not measured on this focused-query set because it has no dimension labels. Fair allocations, section coverage, opposing positions with conditions, and missing dimensions are covered by deterministic tests, not claimed as corpus benchmark results.

Selected settings remain the compatibility baseline: `RERANKER_MIN_SCORE=0.01`, `RERANKER_SCORE_TRANSFORM=model-default`, `PAPER_CAP_POLICY=baseline`, `EVIDENCE_RECOVERY=false`.
The compatibility baseline had a whole-anchor context-budget failure for the bias-definition case in this run; the task-aware arm admitted whole anchors within budget and avoided that failure. Recall includes failed cases, and the report preserves their error messages.
There is no evidence supporting a calibrated new default. Higher page recall can bring more irrelevant passages or unsupported answers. Generation sampling and machine load vary; these single-pass latency and answer-support results have no confidence intervals. Thresholding also affects MMR selection, so locator recall need not change monotonically.

The semantic verifier uses the generation model and can err. Its approval is distinct from factual correctness and answer completeness. JSON reports preserve generated answers, answer digests, verdicts, errors, retrieval provenance, score contracts, rejected candidates, allocations, and recovery queries for human review.

Remaining work: manually review relevance examples including difficult topical negatives and out-of-corpus questions in both query-disjoint splits; evaluate summaries/comparisons/conflicts with section, dimension, position, and study-condition labels; run blind answer review and repeated held-out comparisons before promoting settings.

No index migration or rebuild is needed. Custom rerankers and activation overrides require an explicit model-specific threshold or compatible calibration artifact. Overlapping recovery blocks are conservatively skipped to preserve original evidence IDs; new expanded blocks are conservatively charged to the remaining expansion allowance.

Test validation recorded separately: {'passed': 287, 'warnings': 3}.
