"""Query-local evidence references and deterministic citation checks."""
from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    chunk_id: str | None
    document_id: str | None
    title: str | None
    source: str | None
    page: object | None
    text: str
    scores: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def reference(self) -> str:
        label = self.title or self.source or "Unknown document"
        if self.source and self.source != label:
            label = f"{label} ({self.source})"
        if self.page is not None:
            label += f", page {self.page}"
        return label


@dataclass
class CitationValidation:
    references_valid: bool
    cited_evidence_ids: list[str]
    invalid_evidence_ids: list[str]
    errors: list[str]
    valid_evidence: list[Evidence]
    coverage_warnings: list[str] = field(default_factory=list)
    semantic_support: str = "not_checked"


@dataclass
class GenerationResult:
    answer: str
    original_answer: str
    evidence: list[Evidence]
    validation: CitationValidation
    regeneration_attempts: int
    error_messages: list[str] = field(default_factory=list)


_CITATION = re.compile(r"(?<![\w])\[(E\d+)\](?!\s*\()")
_E_LIKE = re.compile(r"\[(E\s*\d+[^\]]*)\]", re.IGNORECASE)
_LEGACY_CITATION = re.compile(r"\[[^\]\n,]+,\s*(?:[^\]]+,\s*)?p\.\s*\d+[^\]]*\]", re.IGNORECASE)


def assign_evidence(chunks: list[dict]) -> list[Evidence]:
    """Deduplicate by persistent chunk ID (when present), preserving rank."""
    result = []
    seen = set()
    for chunk in chunks:
        chunk_id = chunk.get("id", chunk.get("chunk_id"))
        key = ("id", str(chunk_id)) if chunk_id is not None else ("text", chunk.get("text", ""), chunk.get("source"), chunk.get("page"))
        if key in seen:
            continue
        seen.add(key)
        metadata = dict(chunk.get("metadata") or {})
        scores = {k: v for k, v in chunk.items() if k.endswith(("score", "distance")) and v is not None}
        result.append(Evidence(
            evidence_id=f"E{len(result) + 1}", chunk_id=str(chunk_id) if chunk_id is not None else None,
            document_id=metadata.get("document_id", metadata.get("file_hash")),
            title=chunk.get("title", metadata.get("title")), source=chunk.get("source", metadata.get("source")),
            page=chunk.get("page", metadata.get("page")), text=chunk.get("text", ""),
            scores=scores, metadata=metadata,
        ))
    return result


def check_coverage(answer: str) -> list[str]:
    """Warn on prose sentences without exact evidence references; heuristic only."""
    warnings = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        text = sentence.strip(" \t#>*-`")
        if not text or text.startswith(("[", "|")) or text.endswith((':',)):
            continue
        if re.search(r"\b(Evidence|References|Answer|Note|Not covered)\b", text, re.I) and len(text.split()) < 7:
            continue
        if re.search(r"\b(is|are|was|were|has|have|had|shows?|found|demonstrates?|reported|increased|decreased|improved|reduces?|reduced|uses?|used|includes?)\b", text, re.I) and not _CITATION.search(text):
            warnings.append(text)
    return warnings


def validate_citations(answer: str, evidence: list[Evidence], coverage: bool = True) -> CitationValidation:
    mapping = {item.evidence_id: item for item in evidence}
    cited = _CITATION.findall(answer)
    invalid = []
    errors = []
    for match in _E_LIKE.findall(answer):
        if not re.fullmatch(r"E\d+", match):
            errors.append(f"Malformed evidence citation: [{match}]")
    for match in _LEGACY_CITATION.findall(answer):
        errors.append(f"Unsupported citation format (use [E<number>]): {match}")
    unknown = list(dict.fromkeys(item for item in cited if item not in mapping))
    invalid.extend(unknown)
    if unknown:
        errors.append("Unknown evidence IDs: " + ", ".join(f"[{item}]" for item in unknown))
    duplicates = list(dict.fromkeys(item for item in cited if cited.count(item) > 1))
    if duplicates:
        errors.append("Duplicate evidence citations: " + ", ".join(f"[{item}]" for item in duplicates))
    if not cited:
        errors.append("No evidence citations were found.")
    valid_ids = list(dict.fromkeys(item for item in cited if item in mapping))
    warnings = check_coverage(answer) if coverage else []
    malformed = any("Malformed" in e or "Unsupported citation" in e for e in errors)
    # Reusing a real evidence ID across distinct claims is not a fabricated
    # reference. Keep reporting it, but don't fail the entire response for it.
    return CitationValidation(not invalid and not malformed and bool(cited), cited,
                              invalid, errors, [mapping[item] for item in valid_ids], warnings)
