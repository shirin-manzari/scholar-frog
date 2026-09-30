"""Repeatable offline evaluation for Scholar Frog retrieval and generation."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

from src.citations import GenerationStatus
from src.generate import generate_answer
from src.retrieve import retrieve


DEFAULT_DATASET = Path("evaluations/golden-v1.json")


def load_dataset(path: str | Path) -> dict:
    """Load and validate the deliberately small, human-curated golden set."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read evaluation dataset {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Evaluation dataset must be a version 1 JSON object.")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must contain at least one case.")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
            raise ValueError("Every evaluation case needs a non-empty string id.")
        if case["id"] in ids:
            raise ValueError(f"Evaluation case ID is duplicated: {case['id']}")
        ids.add(case["id"])
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"Evaluation case {case['id']} needs a question.")
        expected = case.get("expected")
        if expected not in {"answered", "abstained"}:
            raise ValueError(f"Evaluation case {case['id']} expected must be answered or abstained.")
        evidence = case.get("evidence", [])
        if not isinstance(evidence, list):
            raise ValueError(f"Evaluation case {case['id']} evidence must be a list.")
        if expected == "answered" and not evidence:
            raise ValueError(f"Answered case {case['id']} needs at least one expected passage.")
        if expected == "abstained" and evidence:
            raise ValueError(f"Abstained case {case['id']} must not specify evidence.")
        for locator in evidence:
            if (not isinstance(locator, dict) or not isinstance(locator.get("source"), str)
                    or not locator["source"] or not isinstance(locator.get("page"), int)
                    or locator["page"] <= 0):
                raise ValueError(
                    f"Evaluation case {case['id']} has an invalid evidence locator."
                )
    return data


def _locator_matches(item, expected: list[dict]) -> bool:
    source = getattr(item, "source", None) if not isinstance(item, dict) else item.get("source")
    page = getattr(item, "page", None) if not isinstance(item, dict) else item.get("page")
    return any(source == locator["source"] and page == locator["page"] for locator in expected)


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def evaluate_dataset(
    dataset_path: str | Path = DEFAULT_DATASET, *, top_k: int = 5,
    retrieval_mode: str | None = None, include_generation: bool = False,
    retrieve_fn: Callable = retrieve, generate_fn: Callable = generate_answer,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Evaluate retrieval always, and generation only when explicitly requested.

    Expected evidence is intentionally a source/page locator, rather than a
    semantic label inferred by a model. This makes recall and citation-locator
    precision reproducible and makes each golden-set change reviewable.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    dataset = load_dataset(dataset_path)
    cases = []
    retrieval_total = retrieval_hits = 0
    retrieval_seconds = 0.0
    generation_total = generation_correct = 0
    generation_seconds = 0.0
    citation_total = citation_correct = citation_valid_answers = 0
    per_paper = defaultdict(lambda: {"expected_cases": 0, "retrieval_hits": 0})

    for case in dataset["cases"]:
        expected = case.get("evidence", [])
        for source in {item["source"] for item in expected}:
            per_paper[source]["expected_cases"] += 1
        started = clock()
        try:
            chunks = retrieve_fn(case["question"], top_k=top_k, retrieval_mode=retrieval_mode)
            retrieval_error = None
        except Exception as exc:  # Record a failed case while preserving the rest of a benchmark run.
            chunks, retrieval_error = [], str(exc)
        elapsed = clock() - started
        retrieval_seconds += elapsed
        retrieval_hit = bool(expected) and any(_locator_matches(chunk, expected) for chunk in chunks)
        if expected:
            retrieval_total += 1
            retrieval_hits += retrieval_hit
            if retrieval_hit:
                for source in {item["source"] for item in expected}:
                    per_paper[source]["retrieval_hits"] += 1

        result = {
            "id": case["id"], "question": case["question"], "expected": case["expected"],
            "expected_evidence": expected, "retrieved": [
                {"id": chunk.get("id"), "source": chunk.get("source"), "page": chunk.get("page")}
                for chunk in chunks
            ],
            "retrieval_hit": retrieval_hit, "retrieval_latency_ms": round(elapsed * 1000, 2),
        }
        if retrieval_error:
            result["retrieval_error"] = retrieval_error

        if include_generation:
            started = clock()
            try:
                generated = generate_fn(case["question"], chunks)
                generation_error = None
            except Exception as exc:
                generated, generation_error = None, str(exc)
            elapsed = clock() - started
            generation_seconds += elapsed
            generation_total += 1
            if generated is not None:
                actual = generated.status.value
                outcome_correct = actual == case["expected"]
                generation_correct += outcome_correct
                evidence = generated.validation.valid_evidence
                citation_total += len(evidence)
                citation_correct += sum(_locator_matches(item, expected) for item in evidence)
                citation_valid_answers += int(
                    generated.status is GenerationStatus.ANSWERED
                    and generated.validation.references_valid
                )
                result.update({
                    "actual": actual, "outcome_correct": outcome_correct,
                    "citation_references_valid": generated.validation.references_valid,
                    "cited_evidence": [
                        {"id": item.evidence_id, "source": item.source, "page": item.page}
                        for item in evidence
                    ],
                })
            if generation_error:
                result["generation_error"] = generation_error
            result["generation_latency_ms"] = round(elapsed * 1000, 2)
        cases.append(result)

    paper_coverage = {
        source: {**values, "recall": _rate(values["retrieval_hits"], values["expected_cases"])}
        for source, values in sorted(per_paper.items())
    }
    metrics = {
        "retrieval_cases": retrieval_total,
        f"retrieval_recall_at_{top_k}": _rate(retrieval_hits, retrieval_total),
        "mean_retrieval_latency_ms": round(retrieval_seconds * 1000 / len(cases), 2),
        "per_paper_coverage": paper_coverage,
    }
    if include_generation:
        metrics.update({
            "generation_cases": generation_total,
            "generation_outcome_accuracy": _rate(generation_correct, generation_total),
            "abstention_correctness": _rate(
                sum(case.get("outcome_correct", False) for case in cases if case["expected"] == "abstained"),
                sum(case["expected"] == "abstained" for case in cases),
            ),
            "citation_locator_precision": _rate(citation_correct, citation_total),
            "citation_valid_answer_rate": _rate(
                citation_valid_answers,
                sum(case["expected"] == "answered" for case in cases),
            ),
            "mean_generation_latency_ms": round(generation_seconds * 1000 / generation_total, 2),
        })
    return {
        "dataset": dataset.get("name", Path(dataset_path).name), "dataset_version": dataset["version"],
        "top_k": top_k, "retrieval_mode": retrieval_mode or "environment default",
        "generation_included": include_generation, "metrics": metrics, "cases": cases,
    }


def write_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
