from pathlib import Path

from src.ingest import (
    chunk_sections,
    chunk_text,
    guess_title,
    remove_repeated_margins,
)


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


def test_chunk_sections_splits_on_markdown_headings_and_tracks_preamble():
    chunks = chunk_sections("Abstract text.\n\n# Paper title\n\n## Methods\nSteps here.")

    assert [section for section, _ in chunks] == [
        "Untitled section", "Paper title", "Methods"
    ]
    assert chunks[1][1].startswith("# Paper title")
    assert chunks[2][1].startswith("## Methods")


def test_heading_sections_hard_split_and_keep_overlap_within_size():
    chunks = chunk_sections("abcdefghij", size=6, overlap=2)

    assert [text for _, text in chunks] == ["abcdef", "efghij"]
    assert all(section == "Untitled section" for section, _ in chunks)
    assert all(len(text) <= 6 for _, text in chunks)


def test_chunk_text_preserves_overlap_for_oversized_content():
    assert chunk_text("abcdefghij", size=6, overlap=2) == ["abcdef", "efghij"]


def test_guess_title_prefers_level_one_markdown_heading():
    title = guess_title(Path("fallback.pdf"), "# **A Markdown Paper Title**\n\nAuthors")

    assert title == "A Markdown Paper Title"


def test_guess_title_falls_back_when_no_markdown_heading_exists():
    title = guess_title(Path("fallback.pdf"), "HEADER\nA useful paper title\nAuthors")

    assert title == "A useful paper title"


def test_repeated_headers_and_footers_are_removed_without_losing_page_numbers():
    title = "A Long Academic Paper Title"
    pages = [
        (1, "# **A Long Academic Paper Title**\nAuthors\nOpening text.\nJournal footer 2024"),
        (2, "A Long Academic Paper Title 2Methods begin here.\nDetails.\nJournal footer 2024"),
        (3, "A Long Academic Paper Title 3Results begin here.\nFindings.\nJournal footer 2024"),
    ]

    cleaned = remove_repeated_margins(pages, title)

    assert [page for page, _ in cleaned] == [1, 2, 3]
    assert cleaned[0][1].startswith("# **A Long Academic Paper Title**")
    assert cleaned[1][1].startswith("Methods begin here.")
    assert cleaned[2][1].startswith("Results begin here.")
    assert all("Journal footer" not in text for _, text in cleaned)


def test_standalone_page_numbers_are_removed_only_at_margins():
    pages = [
        (1, "First body mentions 2 internally.\n1"),
        (2, "Second body mentions 1 internally.\n2"),
    ]

    cleaned = remove_repeated_margins(pages)

    assert cleaned == [
        (1, "First body mentions 2 internally."),
        (2, "Second body mentions 1 internally."),
    ]


def test_repeated_journal_page_locators_are_removed_from_body_prefixes():
    pages = [
        (1, "**10**   # A Long Academic Paper Title\nAuthors"),
        (2, "A. Researcher et al.   10:2   Methods begin here."),
        (3, "A Long Academic Paper Title   10:3   Results begin here."),
        (4, "10:4   A Long Academic Paper Title   Discussion begins here."),
    ]

    cleaned = remove_repeated_margins(pages, "A Long Academic Paper Title")

    assert cleaned == [
        (1, "# A Long Academic Paper Title\nAuthors"),
        (2, "Methods begin here."),
        (3, "Results begin here."),
        (4, "Discussion begins here."),
    ]
