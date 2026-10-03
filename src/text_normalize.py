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
_PUBLICATION_METADATA = re.compile(
    r"^(?:authors?[’']?\s+contact\s+information\s*:|"
    r"[∗*†‡]*\s*(?:co[- ]first|corresponding)\s+authors?\s*\.?$|"
    r"permission to make digital or hard copies\b|"
    r"©\s*\d{4}\s+copyright held\b)",
    re.IGNORECASE,
)
_STANDALONE_DOI = re.compile(r"^https?://(?:dx\.)?doi\.org/\S+$", re.IGNORECASE)
_AUTHOR_NAME = re.compile(
    r"^[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ.'’\-]*(?:\s+[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ.'’\-]*){1,6}"
    r"\s*[∗*†‡\d]*\s*,"
)
_INSTITUTION = re.compile(
    r"\b(?:university|institute|college|academy|laboratory|research|hospital)\b",
    re.IGNORECASE,
)
_RESEARCH_PROSE = re.compile(
    r"\b(?:we|our|this|these|is|are|was|were|has|have|shows?|findings?|"
    r"proposes?|compares?|evaluates?|demonstrates?)\b", re.IGNORECASE,
)


def _is_author_affiliation(text: str) -> bool:
    """Recognize uppercase author bylines, without removing institutional prose."""
    return bool(_AUTHOR_NAME.match(text) and _INSTITUTION.search(text)
                and not _RESEARCH_PROSE.search(text))


def _without_publication_metadata(text: str) -> str:
    """Remove labeled metadata blocks, retaining subsequent research paragraphs."""
    kept = []
    metadata_before = False
    for paragraph in re.split(r"\n\s*\n|(?=^#{1,6}\s)", text, flags=re.MULTILINE):
        plain = plain_text_for_display(paragraph)
        if not plain:
            continue
        if _PUBLICATION_METADATA.match(plain) or _is_author_affiliation(plain):
            metadata_before = True
            continue
        # Hide a publisher's DOI only when it follows a removed metadata block.
        if metadata_before and _STANDALONE_DOI.fullmatch(plain):
            continue
        metadata_before = False
        # Some converters put the abstract directly below a byline without a
        # blank line. Remove only recognized bylines in that mixed paragraph.
        lines = [line for line in paragraph.splitlines()
                 if not _is_author_affiliation(plain_text_for_display(line))]
        kept.append("\n".join(lines))
    return "\n\n".join(kept)


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
    """Return readable evidence without publication metadata or bibliography.

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
    return plain_text_for_display(_without_publication_metadata(excerpt))
