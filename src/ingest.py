import hashlib
import math
import re
from collections import Counter
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


def _plain_title(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[*_`#]+", "", text)
    return " ".join(text.split()).strip()


def _strip_running_title(text: str, title: str) -> str:
    words = re.findall(r"[^\W_]+", _plain_title(title), re.UNICODE)
    if len(words) < 4:
        return text
    title_pattern = r"[\W_]+".join(re.escape(word) for word in words)
    match = re.match(
        rf"^\s*(?:[#*_`]+\s*)*{title_pattern}"
        rf"(?:\s*[*_`#]+)*[ \t]*(?:\d+(?::\d+)?)?[ \t]*",
        text,
        re.IGNORECASE,
    )
    if match is None:
        return text
    return text[match.end():].lstrip(" \t")


def _margin_signature(line: str) -> str | None:
    plain = _plain_title(line).casefold()
    if not plain:
        return None
    if re.fullmatch(r"(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?", plain):
        return "<page-number>"
    plain = re.sub(r"\d+", "#", plain)
    plain = " ".join(plain.split())
    if len(plain) > 200 or len(re.findall(r"[^\W\d_]", plain, re.UNICODE)) < 4:
        return None
    return plain


_LEADING_PAGE_LOCATOR = re.compile(
    r"(?<![\d:])(?:\*\*)?(?P<volume>\d{1,4}):(?P<page>\d{1,4})(?:\*\*)?[ \t]+"
)
_LEADING_VOLUME = re.compile(
    r"^\s*(?:\*\*)?(?P<volume>\d{1,4})(?:\*\*)?[ \t]+"
)


def _leading_page_locator(text: str) -> re.Match[str] | None:
    """Find journal locators such as ``10:2`` in an extracted top margin."""
    return _LEADING_PAGE_LOCATOR.search(text[:200])


def remove_repeated_margins(
    pages: list[tuple[int, str]], title: str | None = None
) -> list[tuple[int, str]]:
    """Remove recurring header/footer lines without changing page numbers."""
    if not pages:
        return []

    prepared = []
    for page_index, (page_number, text) in enumerate(pages):
        if page_index and title:
            text = _strip_running_title(text, title)
        prepared.append((page_number, text))

    threshold = max(2, math.ceil(len(prepared) * 0.3))
    locator_sequences: Counter[tuple[str, int]] = Counter()
    for page_number, text in prepared:
        match = _leading_page_locator(text)
        if match is not None:
            locator_sequences[(
                match.group("volume"), int(match.group("page")) - page_number
            )] += 1
    repeated_locator_sequences = {
        sequence for sequence, count in locator_sequences.items() if count >= threshold
    }
    if repeated_locator_sequences:
        repeated_locator_volumes = {volume for volume, _ in repeated_locator_sequences}
        without_locators = []
        for page_number, text in prepared:
            match = _leading_page_locator(text)
            if match is not None and (
                match.group("volume"), int(match.group("page")) - page_number
            ) in repeated_locator_sequences:
                # Everything before the locator is the alternating running
                # header; the body begins immediately after it.
                text = text[match.end():].lstrip(" \t")
                if title and page_number != prepared[0][0]:
                    text = _strip_running_title(text, title)
            elif page_number == prepared[0][0]:
                volume_match = _LEADING_VOLUME.match(text)
                if (
                    volume_match is not None
                    and volume_match.group("volume") in repeated_locator_volumes
                    and "**" in volume_match.group(0)
                ):
                    text = text[volume_match.end():].lstrip(" \t")
            without_locators.append((page_number, text))
        prepared = without_locators

    top_counts: Counter[str] = Counter()
    bottom_counts: Counter[str] = Counter()
    page_lines = []
    for page_number, text in prepared:
        lines = text.splitlines()
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        top = nonempty[:2]
        bottom = nonempty[-2:]
        for signature in {
            value for i in top if (value := _margin_signature(lines[i])) is not None
        }:
            top_counts[signature] += 1
        for signature in {
            value for i in bottom if (value := _margin_signature(lines[i])) is not None
        }:
            bottom_counts[signature] += 1
        page_lines.append((page_number, lines, top, bottom))

    repeated_top = {value for value, count in top_counts.items() if count >= threshold}
    repeated_bottom = {value for value, count in bottom_counts.items() if count >= threshold}
    cleaned = []
    for page_number, lines, top, bottom in page_lines:
        remove = {
            i for i in top if _margin_signature(lines[i]) in repeated_top
        } | {
            i for i in bottom if _margin_signature(lines[i]) in repeated_bottom
        }
        text = "\n".join(line for i, line in enumerate(lines) if i not in remove).strip()
        if text:
            cleaned.append((page_number, text))
    return cleaned


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
                title = guess_title(pdf_path, pages[0][1])
                pages = remove_repeated_margins(pages, title)
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
            title = guess_title(pdf_path, pages[0][1]) if pages else None
            pages = remove_repeated_margins(pages, title)
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


def _collapsed_with_positions(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace while retaining an offset into the extracted page."""
    characters = []
    positions = []
    in_whitespace = False
    for index, char in enumerate(text):
        if char.isspace():
            if characters and not in_whitespace:
                characters.append(" ")
                positions.append(index)
            in_whitespace = True
        else:
            characters.append(char)
            positions.append(index)
            in_whitespace = False
    return "".join(characters).strip(), positions


def chunk_sections_with_locations(
    text: str, size: int | None = None, overlap: int | None = None,
) -> list[tuple[str, str, int | None, int | None]]:
    """Return chunks plus approximate character offsets in their extracted page.

    The offset is into the post-extraction page text, not a PDF rendering
    coordinate. It stays useful when Markdown formatting changes whitespace and
    gives callers a compact, stable passage locator without changing retrieval
    text or chunking behavior.
    """
    page, page_positions = _collapsed_with_positions(text)
    located = []
    previous_start = 0
    for section, chunk in chunk_sections(text, size, overlap):
        needle, _ = _collapsed_with_positions(chunk)
        start = page.find(needle, previous_start) if needle else -1
        if start < 0:
            # Overlap is concatenated without its original paragraph spacing.
            # Locate the first and last substantial portions when the complete
            # chunk cannot be represented as one contiguous Markdown string.
            prefix = needle[:80].rstrip()
            suffix = needle[-80:].lstrip()
            start = page.find(prefix, previous_start) if prefix else -1
            end_index = page.find(suffix, max(start, previous_start)) if suffix else -1
            if start >= 0 and end_index >= start:
                end = end_index + len(suffix)
            else:
                start, end = -1, -1
        else:
            end = start + len(needle)
        if start >= 0 and end > start and end <= len(page_positions):
            character_start = page_positions[start]
            character_end = page_positions[end - 1] + 1
            previous_start = start
        else:
            character_start = character_end = None
        located.append((section, chunk, character_start, character_end))
    return located


def guess_title(pdf_path: Path, first_page_text: str) -> str:
    """Prefer the first level-1 Markdown heading, then a useful text line."""
    for line in first_page_text.splitlines():
        match = re.match(r"^#\s+(.+?)\s*#*\s*$", line.strip())
        if match and match.group(1).strip():
            return _plain_title(match.group(1))
    for line in first_page_text.splitlines():
        line = line.strip()
        if len(line) > 15 and not line.isupper():
            return _plain_title(line)
    return pdf_path.stem


def open_raw_collection():
    """Open the active physical Chroma collection without declaring it ready.

    This is for synchronization and recovery code that must inspect pending
    writes. Retrieval must use ``get_collection`` so incomplete indexes remain
    inaccessible to normal queries.
    """
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

    return collection


def validate_collection_ready(collection):
    """Apply compatibility and embedding checks to an opened collection."""
    metadata = read_metadata(DB_DIR)
    config = get_index_config()
    count = collection.count()
    check_compatibility(config, metadata, collection_count=count)
    if count:
        try:
            sample = collection.get(limit=1, include=["embeddings"])
            dimension = len(sample["embeddings"][0])
        except Exception as exc:
            raise IndexCompatibilityError("incomplete or corrupted", f"Cannot validate stored embedding dimension: {exc}") from exc
        if dimension != config.embedding_dimension:
            raise IndexCompatibilityError("configuration mismatch", f"Stored collection dimension {dimension} does not match configured dimension {config.embedding_dimension}.")
    return collection


def get_collection():
    """Open the active collection only when it is safe for retrieval."""
    collection = open_raw_collection()
    try:
        return validate_collection_ready(collection)
    except IndexCompatibilityError as exc:
        metadata = read_metadata(DB_DIR)
        count = collection.count()
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
                    status = "configuration mismatch" if "configuration" in reason else "incomplete or corrupted"
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


def ingest_folder(papers_dir: str = "papers"):
    from src.sync import sync_library

    sync_library(papers_dir)


def reingest_folder(papers_dir: str = "papers"):
    """Reprocess every PDF, even if its content hash is already in Chroma."""
    from src.sync import sync_library

    sync_library(papers_dir, reindex=True)
