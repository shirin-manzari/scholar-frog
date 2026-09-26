"""Incremental, recoverable synchronization of the paper folder and Chroma."""
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src import ingest


class SyncError(RuntimeError):
    """The paper library could not be safely synchronized."""


@dataclass
class SyncPlan:
    root: Path
    files: list[dict]
    manifest: dict
    stored: dict
    operations: list[str] = field(default_factory=list)
    added: int = 0
    modified: int = 0
    renamed: int = 0
    deleted: int = 0
    unchanged: int = 0
    duplicated: int = 0
    failures: list[str] = field(default_factory=list)
    prepare_hashes: set[str] = field(default_factory=set)


def manifest_path() -> Path:
    return Path(ingest.DB_DIR) / ingest.MANIFEST_NAME


def _read_manifest() -> dict:
    path = manifest_path()
    if not path.exists():
        return {"version": 1, "documents": {}, "pending": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("documents"), dict):
            raise ValueError("unsupported manifest format")
        if not isinstance(data.get("pending", {}), dict):
            raise ValueError("invalid pending document list")
        data.setdefault("pending", {})
        return data
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SyncError(f"Cannot read document manifest {path}: {exc}") from exc


def _scan(root: Path) -> list[dict]:
    if not root.is_dir():
        raise SyncError(f"Paper library directory does not exist or is not accessible: {root}")
    found = []

    def on_error(exc):
        raise SyncError(f"Incomplete paper library scan: {exc}") from exc

    try:
        for directory, _, names in os.walk(root, onerror=on_error):
            for name in sorted(names, key=str.casefold):
                path = Path(directory) / name
                if path.suffix.lower() != ".pdf":
                    continue
                try:
                    stat = path.stat()
                    found.append({
                        "path": path.relative_to(root).as_posix(), "absolute": path,
                        "hash": ingest.file_hash(path), "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    })
                except OSError as exc:
                    raise SyncError(f"Cannot read {path}; refusing an incomplete scan: {exc}") from exc
    except SyncError:
        raise
    except OSError as exc:
        raise SyncError(f"Incomplete paper library scan: {exc}") from exc
    return sorted(found, key=lambda x: x["path"].casefold())


def _read_index(dry_run: bool) -> dict:
    empty = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    if dry_run and not Path(ingest.DB_DIR).exists():
        return empty
    try:
        if dry_run:
            import chromadb
            client = chromadb.PersistentClient(path=ingest.DB_DIR)
            names = client.list_collections()
            names = [getattr(item, "name", item) for item in names]
            if ingest.COLLECTION_NAME not in names:
                return empty
            collection = client.get_collection(ingest.COLLECTION_NAME)
        else:
            collection = ingest.get_collection()
        return collection.get(include=["documents", "metadatas"])
    except Exception as exc:
        raise SyncError(f"Cannot read existing Chroma index: {exc}") from exc


def _group_records(stored: dict, files: list[dict]) -> dict[str, list[dict]]:
    short_hashes: dict[str, list[dict]] = {}
    for item in files:
        short_hashes.setdefault(item["hash"][:16], []).append(item)
    grouped: dict[str, list[dict]] = {}
    ids = stored.get("ids") or []
    docs = stored.get("documents") or []
    metas = stored.get("metadatas") or [{} for _ in ids]
    for chunk_id, text, metadata in zip(ids, docs, metas):
        metadata = metadata or {}
        digest = metadata.get("document_id") or metadata.get("file_hash")
        if digest in short_hashes and len(digest) == 16:
            candidates = [f for f in short_hashes[digest]
                          if f["path"] == metadata.get("source")
                          or Path(f["path"]).name == metadata.get("source")]
            # The original index stored only the first 16 hash characters and
            # a basename. If a file was renamed before manifest migration,
            # adopt it only when those records identify one full content hash.
            if not candidates:
                unique = {f["hash"] for f in short_hashes[digest]}
                if len(unique) == 1:
                    candidates = short_hashes[digest]
            if candidates:
                digest = candidates[0]["hash"]
        if digest:
            grouped.setdefault(digest, []).append(
                {"id": chunk_id, "text": text, "metadata": metadata}
            )
    return grouped


def plan_sync(papers_dir: str = "papers", *, dry_run: bool = False,
              reindex: bool = False) -> SyncPlan:
    root = Path(papers_dir).expanduser().resolve()
    files = _scan(root)
    manifest = _read_manifest()
    stored = _read_index(dry_run)
    groups = _group_records(stored, files)
    old = manifest["documents"]
    by_hash: dict[str, list[dict]] = {}
    for item in files:
        by_hash.setdefault(item["hash"], []).append(item)
    plan = SyncPlan(root, files, manifest, stored)
    pending = manifest.get("pending", {})
    all_old = set(old) | set(groups) | set(pending)
    old_path_hashes = {}
    for old_hash, entry in old.items():
        for old_path in entry.get("paths", []):
            old_path_hashes[old_path] = old_hash
    discovered_paths = {item["path"] for item in files}

    for digest, copies in sorted(by_hash.items()):
        paths = sorted(f["path"] for f in copies)
        entry = old.get(digest, {})
        records = groups.get(digest, [])
        is_empty_entry = bool(entry.get("status") == "empty" and entry.get("chunk_count") == 0)
        actual_ids = {r["id"] for r in records}
        if "chunk_ids" in entry:
            ids_match = set(entry["chunk_ids"]) == actual_ids
        elif "chunk_count" in entry:
            ids_match = entry["chunk_count"] == len(records)
        else:
            ids_match = bool(records) or is_empty_entry
        if digest not in all_old:
            changed_paths = [path for path in paths if path in old_path_hashes]
            if changed_paths:
                plan.modified += 1
                plan.operations.append(f"~ Update {changed_paths[0]} (content changed)")
            else:
                plan.added += 1
                plan.operations.append(f"+ Index {paths[0]}")
            plan.prepare_hashes.add(digest)
            if len(paths) > 1:
                plan.duplicated += len(paths) - 1
        elif digest in pending or reindex or entry.get("status") == "stale" or (
            not records and not is_empty_entry
        ) or not ids_match or (
            entry and entry.get("index_version") != ingest.INDEX_VERSION
        ):
            plan.modified += 1
            plan.prepare_hashes.add(digest)
            reason = "interrupted write" if digest in pending else (
                "forced reindex" if reindex else "index incomplete or config changed"
            )
            plan.operations.append(f"~ Update {paths[0]} ({reason})")
        elif entry and sorted(entry.get("paths", [])) != paths:
            old_paths = set(entry.get("paths", []))
            added_paths = set(paths) - old_paths
            removed_paths = old_paths - set(paths)
            if len(added_paths) == len(removed_paths) == 1 and not (old_paths & set(paths)):
                plan.renamed += 1
                plan.operations.append(f"> Move {next(iter(removed_paths))} -> {next(iter(added_paths))}")
            else:
                plan.duplicated += len(added_paths)
                plan.operations.append(f"> Update source paths ({len(paths)} identical copies): {paths[0]}")
        elif not entry:
            # Adopt records from the older hash-only index without embeddings.
            plan.unchanged += 1
            if len(paths) > 1:
                plan.duplicated += len(paths) - 1
            plan.operations.append(f"= Reconcile existing index: {paths[0]}")
        else:
            plan.unchanged += 1
            if len(paths) > 1:
                plan.duplicated += len(paths) - 1
                plan.operations.append(f"= Keep {len(paths)} copies of {paths[0]}")

    active = set(by_hash)
    for digest in sorted(all_old - active):
        old_paths = old.get(digest, {}).get("paths", [])
        if any(path in discovered_paths for path in old_paths):
            # A different hash is now present at the same path: version
            # replacement, not a user deletion for safeguard purposes.
            plan.operations.append(f"~ Replace indexed version at {old_paths[0]}")
            continue
        plan.deleted += 1
        label = old_paths[0] if old_paths else digest[:12]
        plan.operations.append(f"- Remove {label}")
    return plan


def _prepare(item: dict, digest: str, paths: list[str]) -> dict:
    pages = ingest.extract_pages(item["absolute"])
    if not pages:
        return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    title = ingest.guess_title(item["absolute"], pages[0][1])
    ids, documents, metadatas = [], [], []
    for page, text in pages:
        for index, (section, chunk) in enumerate(ingest.chunk_sections(text)):
            ids.append(f"{digest}-{page}-{index}")
            documents.append(chunk)
            metadatas.append({
                "source": paths[0], "source_paths": json.dumps(paths),
                "title": title, "section": section, "page": int(page),
                "file_hash": digest, "document_id": digest,
            })
    vectors = []
    if documents:
        vectors = ingest.get_embedding_model().encode(
            documents, show_progress_bar=False, normalize_embeddings=True
        ).tolist()
    return {"ids": ids, "documents": documents, "metadatas": metadatas,
            "embeddings": vectors}


def _write_manifest(data: dict):
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        if temporary:
            temporary.unlink(missing_ok=True)
        raise


def _delete_threshold() -> float:
    try:
        value = float(os.getenv("SYNC_DELETE_THRESHOLD", "0.5"))
    except ValueError as exc:
        raise SyncError("SYNC_DELETE_THRESHOLD must be between 0 and 1") from exc
    if not 0 <= value <= 1:
        raise SyncError("SYNC_DELETE_THRESHOLD must be between 0 and 1")
    return value


def sync_library(papers_dir: str = "papers", *, dry_run: bool = False,
                 force: bool = False, reindex: bool = False) -> SyncPlan:
    plan = plan_sync(papers_dir, dry_run=dry_run, reindex=reindex)
    if dry_run:
        return plan

    indexed = set(plan.manifest["documents"]) | set(_group_records(plan.stored, plan.files))
    ratio = plan.deleted / len(indexed) if indexed else 0
    threshold = _delete_threshold()
    if plan.deleted and ratio > threshold and not force:
        raise SyncError(
            f"Sync would remove {plan.deleted} of {len(indexed)} indexed documents "
            f"({ratio:.0%}), above SYNC_DELETE_THRESHOLD={threshold:.0%}. "
            "Review the library and rerun with --force to confirm."
        )

    collection = ingest.get_collection()
    old_groups = _group_records(plan.stored, plan.files)
    by_hash: dict[str, list[dict]] = {}
    for item in plan.files:
        by_hash.setdefault(item["hash"], []).append(item)

    prepared = {}
    failed_paths = set()
    for digest in sorted(plan.prepare_hashes):
        items = by_hash[digest]
        paths = sorted(item["path"] for item in items)
        try:
            prepared[digest] = _prepare(items[0], digest, paths)
        except Exception as exc:
            failed_paths.update(paths)
            plan.failures.append(f"{paths[0]}: {exc}")
            ingest.console.print(f"[red]Could not prepare {paths[0]}:[/red] {exc}")

    # Record intended IDs before touching Chroma. A crash during upsert leaves
    # a durable marker so the next run re-upserts and verifies the full set.
    pending = dict(plan.manifest.get("pending", {}))
    for digest, document in prepared.items():
        pending[digest] = {
            "paths": sorted(item["path"] for item in by_hash[digest]),
            "chunk_ids": document["ids"], "status": "pending",
        }
    if prepared:
        try:
            _write_manifest({
                "version": 1, "index_version": ingest.INDEX_VERSION,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "documents": plan.manifest["documents"], "pending": pending,
            })
        except OSError as exc:
            raise SyncError(f"Could not persist recovery marker before indexing: {exc}") from exc

    # Deterministic IDs make interrupted upserts safe to retry. Old versions
    # remain present until every replacement has been inserted and read back.
    current: dict[str, list[dict]] = {}
    for digest, document in prepared.items():
        try:
            if document["ids"]:
                collection.upsert(
                    ids=document["ids"], documents=document["documents"],
                    embeddings=document["embeddings"], metadatas=document["metadatas"],
                )
                actual = collection.get(ids=document["ids"], include=["documents", "metadatas"])
                if set(actual["ids"]) != set(document["ids"]):
                    raise RuntimeError("Chroma did not return all written chunks")
                actual_docs = dict(zip(actual["ids"], actual["documents"]))
                expected_docs = dict(zip(document["ids"], document["documents"]))
                if actual_docs != expected_docs:
                    raise RuntimeError("Chroma returned document text that differs from the prepared chunks")
                current[digest] = [
                    {"id": chunk_id, "metadata": metadata}
                    for chunk_id, metadata in zip(document["ids"], document["metadatas"])
                ]
            else:
                current[digest] = []
        except Exception as exc:
            failed_paths.update(item["path"] for item in by_hash[digest])
            plan.failures.append(f"{by_hash[digest][0]['path']}: Chroma write failed: {exc}")
            ingest.console.print(f"[red]Chroma write failed:[/red] {exc}")

    # Existing content is reusable when it is already verified in Chroma.
    for digest, items in by_hash.items():
        if digest not in current and digest not in plan.prepare_hashes:
            current[digest] = old_groups.get(digest, [])

    # If a changed PDF failed, retain the previous version for that path and
    # mark it stale. This avoids losing the last successfully indexed content.
    stale_hashes = set()
    for old_hash, entry in plan.manifest["documents"].items():
        if old_hash in current:
            continue
        if set(entry.get("paths", [])) & failed_paths:
            stale_hashes.add(old_hash)
            current[old_hash] = old_groups.get(old_hash, [])

    active = set(current)
    all_groups = dict(old_groups)
    all_groups.update(current)
    timestamp = datetime.now(timezone.utc).isoformat()
    next_documents = {}
    metadata_changed = False

    for digest in sorted(active):
        source_files = by_hash.get(digest, [])
        paths = sorted(item["path"] for item in source_files)
        previous = plan.manifest["documents"].get(digest, {})
        records = all_groups.get(digest, [])
        if digest in current and digest in prepared:
            records = current[digest]
        if digest in stale_hashes:
            paths = previous.get("paths", [])

        # Update citation paths in-place for renames and duplicate copies.
        if records and paths and digest not in prepared:
            ids, metadata = [], []
            needs_update = False
            for row in records:
                updated = dict(row.get("metadata") or {})
                expected = {
                    "source": paths[0], "source_paths": json.dumps(paths),
                    "document_id": digest, "file_hash": digest,
                }
                if any(updated.get(key) != value for key, value in expected.items()):
                    needs_update = True
                updated.update({
                    "source": paths[0], "source_paths": json.dumps(paths),
                    "document_id": digest, "file_hash": digest,
                })
                ids.append(row["id"])
                metadata.append(updated)
            try:
                if needs_update:
                    collection.update(ids=ids, metadatas=metadata)
                    metadata_changed = True
                records = [{"id": i, "metadata": m} for i, m in zip(ids, metadata)]
            except Exception as exc:
                plan.failures.append(f"{paths[0]}: metadata update failed: {exc}")
                previous_paths = previous.get("paths", [])
                paths = previous_paths or paths
                records = old_groups.get(digest, records)

        file_info = source_files[0] if source_files else {}
        chunk_ids = [row["id"] for row in records]
        next_documents[digest] = {
            "document_id": digest, "content_hash": digest, "paths": paths,
            "primary_path": paths[0] if paths else None,
            "file_size": file_info.get("size", previous.get("file_size", 0)),
            "mtime_ns": file_info.get("mtime_ns", previous.get("mtime_ns", 0)),
            "ingested_at": previous.get("ingested_at", timestamp),
            "updated_at": timestamp, "chunk_count": len(chunk_ids),
            "chunk_ids": chunk_ids,
            "status": "stale" if digest in stale_hashes else ("indexed" if chunk_ids else "empty"),
            "index_version": previous.get("index_version", ingest.INDEX_VERSION)
            if digest in stale_hashes else ingest.INDEX_VERSION,
        }

    obsolete = set(old_groups) - active
    for digest in sorted(obsolete):
        try:
            collection.delete(ids=[row["id"] for row in old_groups[digest]])
        except Exception as exc:
            plan.failures.append(f"Could not remove obsolete document {digest[:12]}: {exc}")
            old_entry = plan.manifest["documents"].get(digest, {})
            if old_entry:
                old_entry = dict(old_entry)
                old_entry["status"] = "orphaned"
                next_documents[digest] = old_entry

    pending = {digest: details for digest, details in pending.items()
               if digest not in current and digest in by_hash}
    manifest = {
        "version": 1, "index_version": ingest.INDEX_VERSION,
        "updated_at": timestamp, "documents": next_documents, "pending": pending,
    }
    try:
        _write_manifest(manifest)
    except OSError as exc:
        raise SyncError(f"Index updated but manifest write failed: {exc}") from exc

    if prepared or obsolete or metadata_changed:
        try:
            from src import retrieve
            retrieve._bm25_cache = None
        except ImportError:
            pass
    plan.manifest = manifest
    return plan
