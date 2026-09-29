import json
from types import SimpleNamespace
from urllib.parse import quote

import pytest

import app
from app import upload_filename
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
