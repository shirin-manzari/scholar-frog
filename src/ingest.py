import hashlib
from pathlib import Path

from rich.console import Console

console = Console()

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
DB_DIR = "chroma_db"
COLLECTION_NAME = "papers"
_embedding_model = None


def file_hash(path: Path) -> str:
    """Content hash so we can detect whether a PDF has already been ingested."""
    h = hashlib.sha256()
    with path.open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()[:16]


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
                page_number = metadata.get("page", i) + 1
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


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware chunking: prefer splitting on blank lines, fall back to
    hard splits only if a single paragraph is longer than `size`."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be between zero and size - 1")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    split_groups = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= size:
            buf = f"{buf}\n{para}".strip()
        else:
            if buf:
                chunks.append(buf)
                split_groups.append(None)
            if len(para) > size:
                step = size - overlap
                split_group = len(split_groups)
                split_parts = []
                start = 0
                while start < len(para):
                    split_parts.append(para[start:start + size])
                    if start + size >= len(para):
                        break
                    start += step
                chunks.extend(split_parts)
                split_groups.extend([split_group] * len(split_parts))
                buf = ""
            else:
                buf = para
    if buf:
        chunks.append(buf)
        split_groups.append(None)

    result = []
    for i, chunk in enumerate(chunks):
        if (i > 0 and overlap and
                not (split_groups[i] is not None and split_groups[i] == split_groups[i - 1])):
            chunk = f"{chunks[i - 1][-overlap:]} {chunk}"
        result.append(chunk)
    return result


def guess_title(pdf_path: Path, first_page_text: str) -> str:
    """Best-effort title guess: first non-trivial line of page 1."""
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
    papers_path = Path(papers_dir)
    if not papers_path.exists():
        console.print(f"[red]Folder not found:[/red] {papers_dir}")
        return

    pdfs = sorted(
        (path for path in papers_path.iterdir()
         if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.lower(),
    )
    if not pdfs:
        console.print(f"[yellow]No PDFs found in {papers_dir}/[/yellow]")
        return

    collection = get_collection()
    existing = collection.get(include=["metadatas"])
    existing_hashes = {
        metadata.get("file_hash")
        for metadata in (existing["metadatas"] or [])
        if metadata and metadata.get("file_hash")
    }
    model = None

    for pdf_path in pdfs:
        h = file_hash(pdf_path)
        if h in existing_hashes:
            console.print(f"[dim]Skipping (already ingested): {pdf_path.name}[/dim]")
            continue

        console.print(f"[cyan]Ingesting:[/cyan] {pdf_path.name}")
        pages = extract_pages(pdf_path)
        if not pages:
            collection.delete(where={"source": pdf_path.name})
            console.print(f"  [yellow]No extractable text (scanned PDF?), skipping.[/yellow]")
            continue

        title = guess_title(pdf_path, pages[0][1])

        all_chunks, all_metas, all_ids = [], [], []
        for page_num, text in pages:
            for j, chunk in enumerate(chunk_text(text)):
                all_chunks.append(chunk)
                all_metas.append({
                    "source": pdf_path.name,
                    "title": title,
                    "page": page_num,
                    "file_hash": h,
                })
                all_ids.append(f"{h}-{page_num}-{j}")

        if not all_chunks:
            continue

        if model is None:
            model = get_embedding_model()
        embeddings = model.encode(
            all_chunks, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

        collection.delete(where={"source": pdf_path.name})
        collection.add(
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metas,
        )
        existing_hashes.add(h)
        console.print(f"  [green]Added {len(all_chunks)} chunks.[/green]")

    console.print("[bold green]Ingestion complete.[/bold green]")
