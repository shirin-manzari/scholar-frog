import json
from types import SimpleNamespace
from urllib.parse import quote

import pytest

import app
from app import MISSING_PAPER_MESSAGES, paper_path, upload_filename
from src.citations import GenerationStatus


@pytest.mark.parametrize("name", [
    "Research notes (2025).pdf",
    "O'Brien, Smith & Chen.pdf",
    "论文—方法.pdf",
])
def test_upload_filename_accepts_common_names(name):
    assert upload_filename(quote(name)) == name


@pytest.mark.parametrize("name", [
    "../other.pdf",
    "folder\\other.pdf",
    "bad\x00name.pdf",
    "paper.txt",
    ".pdf",
])
def test_upload_filename_rejects_unsafe_names(name):
    assert upload_filename(quote(name)) is None


def test_library_status_lists_pdfs_including_subfolders(tmp_path, monkeypatch):
    (tmp_path / "z.pdf").write_bytes(b"%PDF-")
    (tmp_path / "Folder").mkdir()
    (tmp_path / "Folder" / "A.PDF").write_bytes(b"%PDF-")
    (tmp_path / "notes.txt").write_text("not a paper")
    monkeypatch.setattr(app, "PAPERS", tmp_path)
    assert app.library_status() == {
        "papers": ["Folder/A.PDF", "z.pdf"],
    }


def test_paper_path_allows_library_pdf_and_rejects_traversal(tmp_path, monkeypatch):
    (tmp_path / "one.pdf").write_bytes(b"%PDF-")
    monkeypatch.setattr(app, "PAPERS", tmp_path)

    assert paper_path("one.pdf") == tmp_path / "one.pdf"
    assert paper_path("../one.pdf") is None
    assert paper_path("missing.pdf") is None


def test_ask_reports_no_papers_without_searching_a_removed_library(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "PAPERS", tmp_path)
    monkeypatch.setattr(app, "retrieve", lambda *args, **kwargs: pytest.fail("searched stale index"))
    monkeypatch.setattr(app, "generate_answer", lambda *args, **kwargs: pytest.fail("generated without papers"))

    result = app.ask_question("What does the deleted paper say?")

    assert result["status"] == "no_papers"
    assert result["answer"] == ""
    assert result["references"] == []


def test_ask_excludes_removed_sources_but_keeps_existing_alias(tmp_path, monkeypatch):
    (tmp_path / "kept.pdf").write_bytes(b"%PDF-")
    monkeypatch.setattr(app, "PAPERS", tmp_path)
    monkeypatch.setattr(app, "retrieve", lambda *args, **kwargs: [
        {"source": "removed.pdf", "metadata": {"source_paths": json.dumps(["removed.pdf"])}},
        {"source": "old-copy.pdf", "metadata": {"source_paths": json.dumps(["old-copy.pdf", "kept.pdf"])}},
    ])
    observed = []

    def fake_generate(question, chunks):
        observed.extend(chunks)
        validation = SimpleNamespace(
            valid_evidence=[], coverage_warnings=[], semantic_support="not_checked"
        )
        return SimpleNamespace(
            status=GenerationStatus.ABSTAINED, answer="No answer", validation=validation
        )

    monkeypatch.setattr(app, "generate_answer", fake_generate)
    app.ask_question("Question?")

    assert [chunk["source"] for chunk in observed] == ["kept.pdf"]
    assert observed[0]["metadata"]["source"] == "kept.pdf"


def test_ask_passes_selected_paper_and_rejects_missing_selection(tmp_path, monkeypatch):
    (tmp_path / "one.pdf").write_bytes(b"%PDF-")
    monkeypatch.setattr(app, "PAPERS", tmp_path)
    observed = []

    def fake_retrieve(question, top_k, paper):
        observed.append((question, paper))
        return []

    monkeypatch.setattr(app, "retrieve", fake_retrieve)
    monkeypatch.setattr(app, "generate_answer", lambda question, chunks: SimpleNamespace(
        status=GenerationStatus.ABSTAINED, answer="No answer",
        validation=SimpleNamespace(
            valid_evidence=[], coverage_warnings=[], semantic_support="not_checked"
        ),
    ))

    app.ask_question("Summarize this paper.", paper="one.pdf")
    assert observed == [("Summarize this paper.", "one.pdf")]
    with pytest.raises(ValueError, match="Paper moved or renamed|That PDF vanished"):
        app.ask_question("Summarize this paper.", paper="missing.pdf")
    assert len(observed) == 1


def test_missing_selected_paper_uses_a_frog_dialogue(tmp_path, monkeypatch):
    (tmp_path / "one.pdf").write_bytes(b"%PDF-")
    monkeypatch.setattr(app, "PAPERS", tmp_path)
    monkeypatch.setattr(app.random, "choice", lambda messages: messages[1])

    with pytest.raises(ValueError, match="wizards"):
        app.ask_question("Summarize this paper.", paper="missing.pdf")
    assert MISSING_PAPER_MESSAGES[1] == "That PDF vanished. i blame the wizards."


def test_ask_returns_semantic_verdicts_for_ui_display(tmp_path, monkeypatch):
    (tmp_path / "one.pdf").write_bytes(b"%PDF-")
    monkeypatch.setattr(app, "PAPERS", tmp_path)
    monkeypatch.setattr(app, "retrieve", lambda *args, **kwargs: [])
    verdicts = [{"claim_id": "C1", "claim": "A claim.", "supported": True,
                 "reason": "Directly stated."}]
    monkeypatch.setattr(app, "generate_answer", lambda question, chunks: SimpleNamespace(
        status=GenerationStatus.ABSTAINED, answer="No answer",
        validation=SimpleNamespace(valid_evidence=[], coverage_warnings=[],
                                   semantic_support="passed", semantic_verdicts=verdicts),
    ))

    assert app.ask_question("Question?")["semantic_verdicts"] == verdicts


def test_ask_returns_passage_location_for_cited_evidence(tmp_path, monkeypatch):
    (tmp_path / "one.pdf").write_bytes(b"%PDF-")
    monkeypatch.setattr(app, "PAPERS", tmp_path)
    monkeypatch.setattr(app, "retrieve", lambda *args, **kwargs: [])
    evidence = SimpleNamespace(
        evidence_id="E1", reference="Paper, page 2", source="one.pdf", page=2,
        text="A **matched** passage.",
        metadata={"chunk_index": 3, "character_start": 14, "character_end": 36},
    )
    monkeypatch.setattr(app, "generate_answer", lambda question, chunks: SimpleNamespace(
        status=GenerationStatus.ANSWERED, answer="Answer [E1].",
        validation=SimpleNamespace(valid_evidence=[evidence], coverage_warnings=[],
                                   semantic_support="not_checked", semantic_verdicts=[]),
    ))

    assert app.ask_question("Question?")["references"] == [{
        "id": "E1", "reference": "Paper, page 2", "source": "one.pdf", "page": 2,
        "text": "A **matched** passage.", "display_text": "A matched passage.", "passage": 4,
        "character_start": 14, "character_end": 36,
    }]


def test_readable_preview_keeps_canonical_cited_text_and_offsets(tmp_path, monkeypatch):
    from src.citations import assign_evidence
    (tmp_path / "one.pdf").write_bytes(b"%PDF-")
    monkeypatch.setattr(app, "PAPERS", tmp_path)
    raw = (
        "ZHICHENG DOU, Renmin University of China, China&#x20;\n\n"
        "JIAXIN MAO\\<sup>†\\</sup> , Renmin University of China, China&#x20;\n\n"
        "We assess six dimensions: **factuality**, _robustness_, fairness, "
        "transparency, accountability, and \\<u>privacy\\</u>.\n\n"
        "∗Co-first authors.&#x20;\n\n†Corresponding authors.&#x20;\n\n"
        "Authors’ Contact Information: researcher@example.com."
    )
    chunks = [{"id": "expanded", "text": raw, "source": "one.pdf", "title": "Paper", "page": 1,
               "metadata": {"character_start": 663, "character_end": 663 + len(raw)}}]
    evidence = assign_evidence(chunks)
    monkeypatch.setattr(app, "retrieve", lambda *args, **kwargs: chunks)
    monkeypatch.setattr(app, "generate_answer", lambda question, supplied: SimpleNamespace(
        status=GenerationStatus.ANSWERED, answer="Six dimensions [E1].",
        validation=SimpleNamespace(valid_evidence=evidence, coverage_warnings=[],
                                   semantic_support="passed", semantic_verdicts=[])))
    reference = app.ask_question("Which dimensions?")["references"][0]
    assert reference["text"] == raw == evidence[0].text
    assert reference["display_text"] == (
        "We assess six dimensions: factuality, robustness, fairness, "
        "transparency, accountability, and privacy."
    )
    assert reference["character_start"] == 663
    assert reference["character_end"] == 663 + len(raw)
