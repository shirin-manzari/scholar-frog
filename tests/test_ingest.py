from pathlib import Path

from src.ingest import chunk_text, guess_title


def test_chunk_text_keeps_short_paragraphs_together():
    text = "First paragraph.\n\nSecond paragraph."

    assert chunk_text(text, size=100, overlap=10) == [
        "First paragraph.\nSecond paragraph."
    ]


def test_chunk_text_splits_an_oversized_paragraph_with_overlap():
    chunks = chunk_text("abcdefghij", size=6, overlap=2)

    assert chunks == ["abcdef", "efghij"]


def test_guess_title_uses_first_non_trivial_non_uppercase_line():
    title = guess_title(Path("fallback.pdf"), "HEADER\nA useful paper title\nAuthors")

    assert title == "A useful paper title"
