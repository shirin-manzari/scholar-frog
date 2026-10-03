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


def token_count(text: str, tokenizer, *, special: bool = False) -> int:
    """Count locally, without tokenizer truncation."""
    return len(tokenizer.encode(text, add_special_tokens=special, truncation=False))


def tokenizer_limits(model=None):
    model = model or get_embedding_model()
    tokenizer = model.tokenizer
    maximum = min(model.max_seq_length, tokenizer.model_max_length)
    special = tokenizer.num_special_tokens_to_add(pair=False)
    if maximum <= special:
        raise ValueError("Embedding model has no usable token capacity")
    return tokenizer, int(maximum - special)


def page_paragraphs(text: str, section: str = "Untitled section",
                    section_id: int = 0) -> tuple[list[dict], str, int]:
    """Exact page offsets; heading occurrences distinguish repeated section names.

    Carry the final section into the next page during ingestion.
    """
    records = []
    # Headings start a paragraph even without a preceding blank line.
    boundary = re.compile(r"\n\s*\n|(?=^#{1,6}\s)", re.MULTILINE)
    start = 0
    spans = []
    for match in boundary.finditer(text):
        if match.start() > start:
            spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    for left, right in spans:
        raw = text[left:right]
        left += len(raw) - len(raw.lstrip())
        right = left + len(raw.strip())
        if left == right:
            continue
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*(?:\n|$)", text[left:right])
        if heading:
            section = heading.group(1).strip()
            section_id += 1
        records.append({"start": left, "end": right, "section": section,
                        "section_id": section_id})
    return records, section, section_id


def _prefix_end(text: str, start: int, end: int, limit: int, tokenizer) -> int:
    # Binary search a character boundary, then validate the actual token count.
    low, high = start + 1, end
    best = start
    while low <= high:
        mid = (low + high) // 2
        if token_count(text[start:mid], tokenizer) <= limit:
            best, low = mid, mid + 1
        else:
            high = mid - 1
    if best == start:
        raise ValueError("Token budget cannot hold a single character")
    return best


def sentence_spans(text: str, left: int, right: int) -> list[tuple[int, int]]:
    """Conservative sentence boundaries, retaining common academic abbreviations."""
    starts = [left]
    for match in re.finditer(r'(?<=[.!?])\s+(?=[A-Z0-9"“])', text[left:right]):
        prefix = text[left:left + match.start()]
        if re.search(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Fig|Eq|et al|e\.g|i\.e|[A-Z])\.$", prefix):
            continue
        starts.append(left + match.end())
    return [(a, a + len(text[a:b].rstrip())) for a, b in zip(starts, starts[1:] + [right])]


def passage_spans(text: str, paragraphs: list[dict], *, size: int | None = None,
                  overlap: int | None = None, tokenizer=None,
                  max_tokens: int | None = None) -> list[dict]:
    config = get_index_config()
    size = config.chunk_size if size is None else size
    overlap = config.chunk_overlap if overlap is None else overlap
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("size must be positive and overlap between zero and size - 1")
    if tokenizer is None:
        tokenizer, max_tokens = tokenizer_limits()
    capacity = min(size, max_tokens if max_tokens is not None else size)
    overlap = min(overlap, capacity - 1)
    units = []
    for pid, para in enumerate(paragraphs):
        left, right = para["start"], para["end"]
        if token_count(text[left:right], tokenizer) <= capacity:
            units.append((left, right, pid, False))
            continue
        for a, b in sentence_spans(text, left, right):
            units.append((a, b, pid, token_count(text[a:b], tokenizer) > capacity))
    passages = []
    pending = []

    def emit(items):
        a, b = items[0][0], items[-1][1]
        para = paragraphs[items[0][2]]
        passages.append({"start": a, "end": b, "section": para["section"],
                         "section_id": para["section_id"],
                         "paragraph_start": items[0][2], "paragraph_end": items[-1][2]})

    for unit in units:
        a, b, pid, hard = unit
        if hard:
            if pending:
                emit(pending)
                pending = []
            while a < b:
                end = _prefix_end(text, a, b, capacity, tokenizer)
                emit([(a, end, pid, True)])
                if end == b:
                    break
                next_start = end
                while next_start > a and token_count(text[next_start - 1:end], tokenizer) <= overlap:
                    next_start -= 1
                a = max(a + 1, next_start)
            continue
        if pending and (paragraphs[pending[0][2]]["section_id"] != paragraphs[pid]["section_id"]
                        or token_count(text[pending[0][0]:b], tokenizer) > capacity):
            emit(pending)
            tail = []
            if paragraphs[pending[-1][2]]["section_id"] == paragraphs[pid]["section_id"]:
                # A whole paragraph may be larger than the overlap target;
                # use its trailing complete sentences rather than dropping all overlap.
                suffix_units = [(left, right, item[2], False) for item in pending
                                for left, right in sentence_spans(text, item[0], item[1])]
                for old in reversed(suffix_units):
                    if (token_count(text[old[0]:pending[-1][1]], tokenizer) > overlap
                            or token_count(text[old[0]:b], tokenizer) > capacity):
                        break
                    tail.insert(0, old)
            pending = tail
        pending.append(unit)
    if pending:
        emit(pending)
    return passages


def chunk_sections_with_locations(text: str, size: int | None = None,
                                  overlap: int | None = None):
    paragraphs, _, _ = page_paragraphs(text)
    return [(p["section"], text[p["start"]:p["end"]], p["start"], p["end"])
            for p in passage_spans(text, paragraphs, size=size, overlap=overlap)]


def chunk_sections(text: str, size: int | None = None, overlap: int | None = None):
    return [(section, chunk) for section, chunk, _, _
            in chunk_sections_with_locations(text, size, overlap)]


def chunk_text(text: str, size: int | None = None, overlap: int | None = None):
    return [chunk for _, chunk in chunk_sections(text, size, overlap)]


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
