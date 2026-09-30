from src.text_normalize import evidence_excerpt_for_display, plain_text_for_display


def test_display_normalization_removes_markdown_html_and_escapes():
    raw = (
        "these concerns, we propose **Trust-RAG Compass**, across six key "
        "dimensions: _factuality_, _robustness_, and <u>privacy</u>. "
        "TRC Bench (Trust-RAG \\<u>Compass Benchmark), regarding the six\\</u> "
        "dimensions."
    )

    assert plain_text_for_display(raw) == (
        "these concerns, we propose Trust-RAG Compass, across six key dimensions: "
        "factuality, robustness, and privacy. TRC Bench (Trust-RAG Compass Benchmark), "
        "regarding the six dimensions."
    )


def test_display_normalization_keeps_link_label_and_removes_extraction_comments():
    raw = "<!-- Start of picture text --> # Heading\nSee [the paper](https://example.com)."

    assert plain_text_for_display(raw) == "Heading See the paper."


def test_evidence_excerpt_hides_trailing_numbered_bibliography_entries():
    raw = (
        "The survey identifies a research gap: extent, trends, and explanations.\n\n"
        "Retrieved from [https://example.com](https://example.com).\n"
        "- [17] Terra Blevins and Luke Zettlemoyer. 2022. Language contamination.\n"
        "- [18] Rexhina Blloshmi et al. 2021. Generating senses and roles."
    )

    assert evidence_excerpt_for_display(raw) == (
        "The survey identifies a research gap: extent, trends, and explanations."
    )


def test_evidence_excerpt_keeps_ordinary_numbered_citations_in_prose():
    assert evidence_excerpt_for_display("This claim follows earlier work [17].") == (
        "This claim follows earlier work [17]."
    )
