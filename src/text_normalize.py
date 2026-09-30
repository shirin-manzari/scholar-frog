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
