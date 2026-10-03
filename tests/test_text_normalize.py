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


def test_contact_metadata_is_removed_without_losing_surrounding_research():
    raw = (
        "The benchmark evaluates 19 models.\n\n"
        "∗Co-first authors.\n\n†Corresponding authors.\n\n"
        "**Authors’ Contact Information:** A. Researcher, author@example.com;\n"
        "B. Researcher, Example University.\n\n"
        "## Methods\nWe compare six dimensions."
    )
    assert evidence_excerpt_for_display(raw) == (
        "The benchmark evaluates 19 models. Methods We compare six dimensions."
    )
    assert "author@example.com" in raw


def test_straight_apostrophe_contact_label_and_publisher_notices_are_hidden():
    raw = (
        "Results remain useful.\n\nAuthors' Contact Information: a@example.com.\n\n"
        "Permission to make digital or hard copies of all or part of this work.\n\n"
        "© 2018 Copyright held by the owner/author(s).\n\n"
        "https://doi.org/XXXXXXX.XXXXXXX"
    )
    assert evidence_excerpt_for_display(raw) == "Results remain useful."


def test_research_about_contacts_copyright_and_doi_links_is_preserved():
    raw = (
        "We study authors’ contact information and corresponding authors.\n\n"
        "Copyright restrictions affect the training corpus.\n\n"
        "https://doi.org/10.1234/research"
    )
    assert evidence_excerpt_for_display(raw) == plain_text_for_display(raw)


def test_contact_block_does_not_hide_a_following_heading_without_blank_line():
    raw = "Authors’ Contact Information: a@example.com.\n## Results\nAccuracy improved."
    assert evidence_excerpt_for_display(raw) == "Results Accuracy improved."


def test_author_affiliations_and_combined_bylines_are_hidden():
    raw = (
        "ZHICHENG DOU, Renmin University of China, China "
        "PHILIP S. YU, University of Illinois, USA\n\n"
        "JIAXIN MAO<sup>†</sup> , Renmin University of China,\nChina\n\n"
        "We evaluate the trustworthiness of RAG systems."
    )
    assert evidence_excerpt_for_display(raw) == (
        "We evaluate the trustworthiness of RAG systems."
    )


def test_abstract_directly_below_an_author_line_is_preserved():
    raw = (
        "YUJIA ZHOU<sup>∗</sup> , Tsinghua University, China\n"
        "Retrieval-Augmented Generation has grown into a pivotal paradigm."
    )
    assert evidence_excerpt_for_display(raw) == (
        "Retrieval-Augmented Generation has grown into a pivotal paradigm."
    )


def test_institution_mentions_and_author_led_research_claims_are_preserved():
    for raw in (
        "Researchers at Renmin University of China evaluate language models.",
        "ZHICHENG DOU, at Renmin University, proposes a new method.",
        "The University of Illinois is included in the study.",
    ):
        assert evidence_excerpt_for_display(raw) == raw
