from urllib.parse import quote

import pytest

import app
from app import upload_filename


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
