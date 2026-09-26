import hashlib
import re
from pathlib import Path

from rich.console import Console

console = Console()

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
DB_DIR = "chroma_db"
COLLECTION_NAME = "papers"
MANIFEST_NAME = "documents.json"
INDEX_VERSION = f"{EMBED_MODEL_NAME}:chunks-{CHUNK_SIZE}-{CHUNK_OVERLAP}:v1"
_embedding_model = None


def file_hash(path: Path) -> str:
    """Content hash so we can detect whether a PDF has already been ingested."""
    h = hashlib.sha256()
    with path.open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def get_embedding_model():
    """Load and cache the local embedding model on first use."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedding_model


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Returns [(page_number, markdown), ...], 1-indexed pages.

    If Markdown conversion fails or produces no usable page text, retain the
    plain-text extraction path so a conversion issue does not drop a PDF.
    """
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        try:
            import pymupdf4llm

            converted = pymupdf4llm.to_markdown(doc, page_chunks=True)
            pages = []
            for i, page_chunk in enumerate(converted or []):
                text = page_chunk.get("text", "") if isinstance(page_chunk, dict) else ""
                metadata = page_chunk.get("metadata", {}) if isinstance(page_chunk, dict) else {}
                page_number = metadata.get("page_number", metadata.get("page", i + 1))
                if text.strip():
                    pages.append((page_number, text))
            if pages:
                console.print(f"  [dim]Extraction: Markdown ({len(pages)} pages)[/dim]")
                return pages
            raise ValueError("Markdown conversion returned no usable page text")
        except Exception as exc:
            console.print(f"  [yellow]Markdown extraction failed ({exc}); using plain-text fallback.[/yellow]")
            pages = []
            for i, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    pages.append((i + 1, text))
            console.print(f"  [dim]Extraction: plain-text fallback ({len(pages)} pages)[/dim]")
            return pages


def chunk_sections(text: str, size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[tuple[str, str]]:
    """Chunk Markdown by heading sections, returning (section, chunk) pairs."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be between zero and size - 1")

    sections: list[tuple[str, list[str]]] = []
    current_title = "Untitled section"
    current_lines: list[str] = []
    heading = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
    for line in text.splitlines():
        match = heading.match(line.strip())
        if match:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = match.group(1).strip()
            current_lines = [line.strip()]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))

    chunks: list[tuple[str, str]] = []
    for section_title, lines in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(lines)) if p.strip()]
        section_chunks = []
        buf = ""
        for para in paragraphs:
            capacity = size if not section_chunks else size - overlap
            if not buf and len(para) <= capacity:
                buf = para
            elif buf and len(buf) + len(para) + 1 <= capacity:
                buf = f"{buf}\n{para}"
            else:
                if buf:
                    section_chunks.append(buf)
                    buf = ""
                capacity = size if not section_chunks else size - overlap
                if len(para) <= capacity:
                    buf = para
                else:
                    start = 0
                    while start < len(para):
                        capacity = size if not section_chunks else size - overlap
                        part = para[start:start + capacity]
                        section_chunks.append(part)
                        start += len(part)
        if buf:
            section_chunks.append(buf)

        for i, chunk in enumerate(section_chunks):
            if i and overlap:
                chunk = f"{section_chunks[i - 1][-overlap:]}{chunk}"
            chunks.append((section_title, chunk))
    return chunks


def chunk_text(text: str, size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Compatibility helper returning heading-aware Markdown chunks."""
    return [chunk for _, chunk in chunk_sections(text, size, overlap)]


def guess_title(pdf_path: Path, first_page_text: str) -> str:
    """Prefer the first level-1 Markdown heading, then a useful text line."""
    for line in first_page_text.splitlines():
        match = re.match(r"^#\s+(.+?)\s*#*\s*$", line.strip())
        if match and match.group(1).strip():
            return match.group(1).strip()
    for line in first_page_text.splitlines():
        line = line.strip()
        if len(line) > 15 and not line.isupper():
            return line
    return pdf_path.stem


def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)


def ingest_folder(papers_dir: str = "papers"):
    from src.sync import sync_library

    sync_library(papers_dir)


def reingest_folder(papers_dir: str = "papers"):
    """Reprocess every PDF, even if its content hash is already in Chroma."""
    from src.sync import sync_library

    sync_library(papers_dir, reindex=True)
