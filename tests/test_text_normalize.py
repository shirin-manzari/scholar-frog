from src.text_normalize import plain_text_for_display


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
