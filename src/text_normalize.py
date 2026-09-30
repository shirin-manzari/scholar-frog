"""Safe display-only cleanup for text extracted from PDF Markdown."""
from __future__ import annotations

import html
import re


_COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)
_TAGS = re.compile(r"</?[A-Za-z][^>]*>")
_MARKDOWN_LINK = re.compile(r"!?\[([^\]]*)\]\([^\s)]+(?:\s+['\"][^)]*['\"])?\)")
_FENCED_CODE = re.compile(r"```(?:[^\n]*)\n?(.*?)```", re.DOTALL)
_HEADING_OR_LIST = re.compile(r"(?m)^\s*(?:#{1,6}\s+|[-+*]\s+|\d+[.)]\s+|>\s?)")
_ESCAPED_MARKDOWN = re.compile(r"\\([\\`*{}\[\]<>_()#+.!-])")
_BIBLIOGRAPHY_HEADING = re.compile(r"(?im)^\s*#{0,6}\s*(?:references|bibliography)\s*$")
_BIBLIOGRAPHY_ENTRY = re.compile(r"(?m)^\s*(?:[-+*]\s*)?\[\d{1,4}\]\s+[A-Z]")
_INLINE_BIBLIOGRAPHY_ENTRY = re.compile(r"\[\d{1,4}\]\s+[A-Z][\w-]+")
_TRAILING_REFERENCE_LINK = re.compile(
    r"\s*(?:retrieved from|doi:)\s*!?\[[^\]]+\]\([^)]*\)\.?\s*$", re.IGNORECASE
)


def plain_text_for_display(text: str) -> str:
    """Convert common PDF-to-Markdown artifacts into readable plain text.

    This intentionally runs only at display time: indexed and LLM context text
    remains untouched, so retrieval ranking and citation validation retain the
    original extracted passage.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _COMMENTS.sub(" ", text)
    text = _FENCED_CODE.sub(lambda match: match.group(1), text)
    text = _MARKDOWN_LINK.sub(lambda match: match.group(1), text)
    # PDF converters sometimes escape their own HTML output (``\<u>``).
    # Unescape before removing tags so both forms disappear consistently.
    text = _ESCAPED_MARKDOWN.sub(r"\1", text)
    text = _TAGS.sub("", text)
    text = text.replace("`", "")
    text = re.sub(r"(?:\*\*|__|~~)(.+?)(?:\*\*|__|~~)", r"\1", text)
    text = re.sub(r"(?<!\w)[*_]([^\n*_]+?)[*_](?!\w)", r"\1", text)
    text = _HEADING_OR_LIST.sub("", text)
    return " ".join(text.split())


def evidence_excerpt_for_display(text: str) -> str:
    """Return a readable evidence excerpt without trailing bibliography entries.

    A retrieval chunk can cross from prose into a reference list. The list stays
    in the indexed chunk for retrieval and generation, but it obscures the
    useful passage in a human-facing evidence card.
    """
    if not isinstance(text, str):
        return ""
    heading = _BIBLIOGRAPHY_HEADING.search(text)
    entry = _BIBLIOGRAPHY_ENTRY.search(text)
    cut_at = min(
        (match.start() for match in (heading, entry) if match is not None),
        default=None,
    )
    if cut_at is None:
        # Some converters collapse bibliography lines. Require two numbered,
        # author-style entries so ordinary inline citations remain intact.
        inline_entries = list(_INLINE_BIBLIOGRAPHY_ENTRY.finditer(text))
        if len(inline_entries) >= 2:
            cut_at = inline_entries[0].start()
    excerpt = text if cut_at is None else text[:cut_at]
    if cut_at is not None:
        excerpt = _TRAILING_REFERENCE_LINK.sub("", excerpt)
    return plain_text_for_display(excerpt)
