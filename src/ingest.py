import hashlib
import re
from pathlib import Path

from rich.console import Console
from src.index_config import (IndexCompatibilityError, check_compatibility,
                              load_index_config, read_metadata)

console = Console()

DB_DIR = "chroma_db"
COLLECTION_NAME = "papers"
MANIFEST_NAME = "documents.json"
_embedding_model = None
_embedding_model_key = None


def get_index_config(**overrides):
    return load_index_config(overrides=overrides)


def active_collection_name():
    metadata = read_metadata(DB_DIR)
    return (metadata or {}).get("active_collection", COLLECTION_NAME)


def index_status(*, config=None):
    config = config or get_index_config()
    metadata = read_metadata(DB_DIR)
    count = 0
    try:
        import chromadb
        client = chromadb.PersistentClient(path=DB_DIR)
        names = [getattr(item, "name", item) for item in client.list_collections()]
        name = (metadata or {}).get("active_collection", COLLECTION_NAME)
        if name not in names and metadata is not None:
            raise IndexCompatibilityError("incomplete or corrupted", f"Active Chroma collection {name!r} is missing.")
        if name in names:
            count = client.get_collection(name).count()
    except IndexCompatibilityError:
        raise
    except Exception:
        if metadata is not None:
            raise IndexCompatibilityError("incomplete or corrupted", "Cannot inspect the active Chroma collection.")
    try:
        status = check_compatibility(config, metadata, collection_count=count)
    except IndexCompatibilityError as exc:
        if metadata is None and count:
            # Ask the sync layer whether this is the narrowly defined,
            # manifest-backed first-write recovery state. Retrieval itself
            # still remains blocked by get_collection's compatibility check.
            try:
                from src.sync import _initial_recovery_state, _read_manifest
                physical = client.get_collection(name).get(include=["documents", "metadatas"])
                manifest = _read_manifest()
                recoverable, reason = _initial_recovery_state(manifest, physical, metadata)
                if recoverable:
                    return "recoverable_pending_initial", metadata, count
                if manifest.get("pending") and "configuration" in reason:
                    return "incompatible", metadata, count
                if manifest.get("pending") and "unknown" in reason:
                    return "corrupted", metadata, count
            except Exception:
                pass
        raise exc
    return status, metadata, count


def file_hash(path: Path) -> str:
    """Content hash so we can detect whether a PDF has already been ingested."""
    h = hashlib.sha256()
    with path.open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def get_embedding_model():
    """Load and cache the local embedding model on first use."""
    global _embedding_model, _embedding_model_key
    config = get_index_config()
    key = (config.embedding_model, config.embedding_revision)
    if _embedding_model is None or _embedding_model_key != key:
        from sentence_transformers import SentenceTransformer
        kwargs = {"revision": config.embedding_revision} if config.embedding_revision else {}
        _embedding_model = SentenceTransformer(config.embedding_model, **kwargs)
        actual_dimension = _embedding_model.get_sentence_embedding_dimension()
        if actual_dimension != config.embedding_dimension:
            raise IndexCompatibilityError(
                "configuration mismatch",
                f"Embedding model {config.embedding_model!r} returns dimension {actual_dimension}; configured dimension is {config.embedding_dimension}.",
            )
        _embedding_model_key = key
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


def chunk_sections(text: str, size: int | None = None,
                   overlap: int | None = None) -> list[tuple[str, str]]:
    """Chunk Markdown by heading sections, returning (section, chunk) pairs."""
    config = get_index_config()
    size = config.chunk_size if size is None else size
    overlap = config.chunk_overlap if overlap is None else overlap
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


def chunk_text(text: str, size: int | None = None,
               overlap: int | None = None) -> list[str]:
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
    name = active_collection_name()
    metadata = read_metadata(DB_DIR)
    names = [getattr(item, "name", item) for item in client.list_collections()]
    if name in names:
        collection = client.get_collection(name)
    elif metadata is not None:
        raise IndexCompatibilityError("incomplete or corrupted", f"Active Chroma collection {name!r} is missing.")
    else:
        collection = client.get_or_create_collection(name)
    config = get_index_config()
    count = collection.count()
    try:
        check_compatibility(config, metadata, collection_count=count)
    except IndexCompatibilityError as exc:
        if metadata is None and count:
            try:
                from src.sync import _initial_recovery_state, _read_manifest
                manifest = _read_manifest()
                if manifest.get("pending"):
                    physical = collection.get(include=["documents", "metadatas"])
                    recoverable, reason = _initial_recovery_state(manifest, physical, metadata)
                    if recoverable:
                        raise IndexCompatibilityError(
                            "recovering", "The initial index is incomplete and awaiting recovery. "
                            "Run `python ask.py sync` before asking questions."
                        ) from exc
                    status = "configuration mismatch" if "configuration differs" in reason else "incomplete or corrupted"
                    raise IndexCompatibilityError(
                        status,
                        f"The initial index is incomplete and cannot be recovered safely: {reason}. "
                        "Run `python ask.py index rebuild` or clear the incomplete index.",
                    ) from exc
            except IndexCompatibilityError:
                raise
            except Exception:
                pass
        raise
    if count:
        try:
            sample = collection.get(limit=1, include=["embeddings"])
            dimension = len(sample["embeddings"][0])
        except Exception as exc:
            raise IndexCompatibilityError("incomplete or corrupted", f"Cannot validate stored embedding dimension: {exc}") from exc
        if dimension != config.embedding_dimension:
            raise IndexCompatibilityError("configuration mismatch", f"Stored collection dimension {dimension} does not match configured dimension {config.embedding_dimension}.")
    return collection


def ingest_folder(papers_dir: str = "papers"):
    from src.sync import sync_library

    sync_library(papers_dir)


def reingest_folder(papers_dir: str = "papers"):
    """Reprocess every PDF, even if its content hash is already in Chroma."""
    from src.sync import sync_library

    sync_library(papers_dir, reindex=True)
