"""Incremental, recoverable synchronization of the paper folder and Chroma."""
import json
import hashlib
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src import ingest
from src.index_config import (INDEX_SCHEMA_VERSION, IndexCompatibilityError, check_compatibility,
                              new_metadata, read_metadata, write_metadata)


class SyncError(RuntimeError):
    """The paper library could not be safely synchronized."""


INDEX_LOCK = threading.RLock()
_snapshot_cache: dict[str, tuple[tuple[int, int, int, str], "CommittedSnapshot"]] = {}


@dataclass(frozen=True)
class CommittedSnapshot:
    revision: str
    chunk_ids: frozenset[str]
    owners: dict[str, dict]


def _committed_revision(documents: dict, config_fingerprint: str) -> str:
    revision_rows = []
    for digest, entry in sorted(documents.items()):
        if entry.get("status") not in {"indexed", "stale"}:
            continue
        ids = entry.get("chunk_ids")
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise IndexCompatibilityError(
                "incomplete or corrupted",
                f"Committed document {digest[:12]} has no valid committed chunk ID list.",
            )
        revision_rows.append((digest, ids, entry.get("paths", [])))
    revision_json = json.dumps(
        [config_fingerprint, revision_rows], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(revision_json.encode()).hexdigest()


def committed_snapshot(*, collection_count: int = 0) -> CommittedSnapshot:
    """Return the committed manifest view; pending and orphan records are excluded."""
    path = manifest_path()
    if not path.exists():
        if collection_count:
            raise IndexCompatibilityError(
                "incomplete or corrupted",
                f"The active Chroma collection has {collection_count} chunks but its document manifest is missing. "
                "Restore the manifest or rebuild the index before retrieval.",
            )
        return CommittedSnapshot("empty", frozenset(), {})
    try:
        stat = path.stat()
        config_fingerprint = ingest.get_index_config().fingerprint
        signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino, config_fingerprint)
        with INDEX_LOCK:
            cached = _snapshot_cache.get(str(path))
            if cached and cached[0] == signature:
                return cached[1]
            data = _read_manifest(path=path)
            if data.get("index_version") != config_fingerprint:
                raise IndexCompatibilityError(
                    "configuration mismatch",
                    "The document manifest fingerprint does not match the active index configuration; rebuild required.",
                )
            owners = {}
            revision_rows = []
            for digest, entry in sorted(data["documents"].items()):
                if entry.get("status") not in {"indexed", "stale"}:
                    continue
                ids = entry.get("chunk_ids")
                if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
                    raise IndexCompatibilityError(
                        "incomplete or corrupted",
                        f"Committed document {digest[:12]} has no valid committed chunk ID list.",
                    )
                paths = entry.get("paths", [])
                owner = {"document_id": digest, "paths": paths}
                for chunk_id in ids:
                    if chunk_id in owners:
                        raise IndexCompatibilityError(
                            "incomplete or corrupted",
                            f"Chunk ID {chunk_id!r} is referenced by multiple committed documents.",
                        )
                    owners[chunk_id] = owner
                revision_rows.append((digest, ids, paths))
            revision_json = json.dumps(
                [config_fingerprint, revision_rows], sort_keys=True, separators=(",", ":")
            )
            revision = hashlib.sha256(revision_json.encode()).hexdigest()
            snapshot = CommittedSnapshot(
                revision,
                frozenset(owners), owners,
            )
            if cached and cached[1].revision == revision:
                _snapshot_cache[str(path)] = (signature, cached[1])
                return cached[1]
            _snapshot_cache[str(path)] = (signature, snapshot)
            return snapshot
    except IndexCompatibilityError:
        raise
    except (SyncError, OSError, ValueError, TypeError, KeyError) as exc:
        raise IndexCompatibilityError(
            "incomplete or corrupted", f"Cannot load committed document manifest {path}: {exc}"
        ) from exc


def _invalidate_snapshot(path: Path | None = None):
    with INDEX_LOCK:
        if path is None:
            _snapshot_cache.clear()
        else:
            _snapshot_cache.pop(str(path), None)


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
    succeeded: int = 0
    failures: list[str] = field(default_factory=list)
    prepare_hashes: set[str] = field(default_factory=set)


def manifest_path(collection_name: str | None = None) -> Path:
    root = Path(ingest.DB_DIR)
    if collection_name:
        metadata = read_metadata(ingest.DB_DIR)
        active = (metadata or {}).get("active_collection", ingest.COLLECTION_NAME)
        if collection_name == active:
            filename = (metadata or {}).get("manifest_name", ingest.MANIFEST_NAME)
        else:
            filename = f"documents.{collection_name}.json"
        return root / filename
    metadata = read_metadata(ingest.DB_DIR)
    return root / (metadata or {}).get("manifest_name", ingest.MANIFEST_NAME)


def _read_manifest(collection_name: str | None = None, *, path: Path | None = None) -> dict:
    path = path or manifest_path(collection_name)
    if not path.exists():
        return {"version": 1, "documents": {}, "pending": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("manifest root must be an object")
        if data.get("version") != 1 or not isinstance(data.get("documents"), dict):
            raise ValueError("unsupported manifest format")
        if not isinstance(data.get("pending", {}), dict):
            raise ValueError("invalid pending document list")
        data.setdefault("pending", {})
        return data
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
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


def _read_index(dry_run: bool, collection_name: str | None = None) -> dict:
    empty = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    if dry_run and not Path(ingest.DB_DIR).exists():
        return empty
    try:
        if dry_run:
            import chromadb
            client = chromadb.PersistentClient(path=ingest.DB_DIR)
            names = client.list_collections()
            names = [getattr(item, "name", item) for item in names]
            name = collection_name or ingest.active_collection_name()
            if name not in names:
                return empty
            collection = client.get_collection(name)
        else:
            if collection_name:
                import chromadb
                client = chromadb.PersistentClient(path=ingest.DB_DIR)
                collection = client.get_or_create_collection(collection_name)
            else:
                collection = _collection_for_sync()
        return collection.get(include=["documents", "metadatas"])
    except Exception as exc:
        raise SyncError(f"Cannot read existing Chroma index: {exc}") from exc


def _collection_for_sync():
    """Open the physical collection for sync, including a pending first write.

    Retrieval continues to use ``ingest.get_collection`` and its strict
    readiness checks. Synchronization needs to inspect an unversioned
    collection before deciding whether its rows are a safe pending write.
    """
    try:
        return ingest.get_collection()
    except IndexCompatibilityError as exc:
        if exc.status not in {"legacy or unversioned", "legacy"}:
            raise
        import chromadb
        client = chromadb.PersistentClient(path=ingest.DB_DIR)
        return client.get_or_create_collection(ingest.active_collection_name())


def _initial_recovery_state(manifest: dict, stored: dict, metadata: dict | None) -> tuple[bool, str]:
    """Classify the narrow unversioned, interrupted initial-index case."""
    stored_ids = list(stored.get("ids") or [])
    ids = set(stored_ids)
    if metadata is not None:
        return False, "index metadata already exists"
    if not ids:
        return False, "collection has no physical chunks"
    pending = manifest.get("pending")
    if not isinstance(pending, dict) or not pending:
        return False, "no trustworthy pending document writes exist"
    config = ingest.get_index_config()
    fingerprint = config.fingerprint
    if manifest.get("index_version") != fingerprint:
        return False, "pending index configuration differs; run `python ask.py index rebuild` or clear the incomplete index"
    if manifest.get("index_schema_version", INDEX_SCHEMA_VERSION) != INDEX_SCHEMA_VERSION:
        return False, "pending index schema differs; run `python ask.py index rebuild` or clear the incomplete index"
    intended_config = manifest.get("index_configuration")
    if intended_config is not None and intended_config != config.canonical():
        return False, "pending index configuration differs; run `python ask.py index rebuild` or clear the incomplete index"

    pending_ids: set[str] = set()
    owner_by_id = {}
    committed_ids: set[str] = set()
    committed_owner = {}
    for digest, entry in manifest.get("documents", {}).items():
        if entry.get("status") not in {"indexed", "stale"}:
            continue
        chunk_ids = entry.get("chunk_ids")
        if (not isinstance(chunk_ids, list)
                or any(not isinstance(value, str) or not value for value in chunk_ids)
                or len(set(chunk_ids)) != len(chunk_ids)):
            return False, f"committed document {digest[:12]} has malformed chunk IDs"
        committed_ids.update(chunk_ids)
        for chunk_id in chunk_ids:
            if chunk_id in committed_owner:
                return False, f"committed chunk {chunk_id!r} is claimed by multiple documents"
            committed_owner[chunk_id] = digest
    for digest, entry in pending.items():
        if (not isinstance(entry, dict) or entry.get("status") != "pending"
                or not isinstance(entry.get("chunk_ids"), list)
                or not isinstance(entry.get("paths"), list) or not entry.get("paths")
                or any(not isinstance(path, str) or not path for path in entry["paths"])
                or any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in entry["chunk_ids"])
                or len(set(entry["chunk_ids"])) != len(entry["chunk_ids"])):
            continue
        if entry.get("configuration_fingerprint", fingerprint) != fingerprint:
            return False, f"pending document {digest[:12]} was written with a different configuration; rebuild or clear the incomplete index"
        if entry.get("configuration") not in (None, config.canonical()):
            return False, f"pending document {digest[:12]} was written with a different configuration; rebuild or clear the incomplete index"
        for chunk_id in entry["chunk_ids"]:
            if chunk_id in owner_by_id:
                return False, f"pending chunk {chunk_id!r} is claimed by multiple document writes"
            owner_by_id[chunk_id] = digest
            pending_ids.add(chunk_id)

    known_ids = pending_ids | committed_ids
    unknown = ids - known_ids
    if unknown:
        return False, f"physical chunks are not referenced by pending writes ({len(unknown)} unknown); rebuild or clear the incomplete index"
    if not committed_ids.issubset(ids):
        return False, "committed manifest versions are missing physical chunks; rebuild or clear the incomplete index"
    if not pending_ids or not ids:
        return False, "pending records do not identify the physical initial write"

    physical = dict(zip(
        stored_ids,
        zip(stored.get("documents") or [], stored.get("metadatas") or []),
    ))
    for chunk_id, (text, raw_metadata) in physical.items():
        digest = owner_by_id.get(chunk_id)
        metadata = raw_metadata or {}
        expected_owner = digest or committed_owner.get(chunk_id)
        if expected_owner is not None and (
            metadata.get("document_id") != expected_owner or metadata.get("file_hash") != expected_owner
        ):
            return False, f"physical chunk {chunk_id!r} does not match its pending document; rebuild or clear the incomplete index"
    return True, "pending initial index matches the active configuration and physical chunks"


def _verify_pending_embeddings(collection, stored: dict) -> None:
    """Check stored vector dimensions before rewriting a pending initial version."""
    ids = list(stored.get("ids") or [])
    if not ids:
        return
    try:
        result = collection.get(ids=ids, include=["embeddings"])
        embeddings = result.get("embeddings")
    except Exception as exc:
        raise IndexCompatibilityError(
            "incomplete or corrupted", f"Cannot inspect embeddings for pending initial chunks: {exc}"
        ) from exc
    if embeddings is None:
        return
    dimension = ingest.get_index_config().embedding_dimension
    if (len(embeddings) != len(ids)
            or any(vector is None or len(vector) != dimension for vector in embeddings)):
        raise IndexCompatibilityError(
            "configuration mismatch",
            f"Pending initial chunks do not match the configured embedding dimension {dimension}; "
            "run `python ask.py index rebuild` or clear the incomplete index.",
        )


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


def _committed_groups(groups: dict[str, list[dict]], manifest: dict) -> dict[str, list[dict]]:
    """Filter physical Chroma rows to the manifest's committed version IDs."""
    committed = {}
    for digest, entry in manifest.get("documents", {}).items():
        if entry.get("status") not in {"indexed", "stale"}:
            continue
        ids = set(entry.get("chunk_ids", []))
        if ids:
            committed[digest] = [row for row in groups.get(digest, []) if row["id"] in ids]
    return committed


def plan_sync(papers_dir: str = "papers", *, dry_run: bool = False,
              reindex: bool = False, collection_name: str | None = None) -> SyncPlan:
    root = Path(papers_dir).expanduser().resolve()
    files = _scan(root)
    manifest = _read_manifest(collection_name)
    stored = _read_index(dry_run, collection_name)
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
        elif digest in pending or reindex or entry.get("status") in {"stale", "orphaned"} or (
            not records and not is_empty_entry
        ) or not ids_match or (
            entry and entry.get("configuration_fingerprint") != ingest.get_index_config().fingerprint
        ):
            plan.modified += 1
            plan.prepare_hashes.add(digest)
            reason = "interrupted write" if digest in pending else (
                "forced reindex" if reindex else "index incomplete or config changed"
            )
            plan.operations.append(f"~ Update {paths[0]} ({reason})")
        elif records and not entry:
            # Rows lacking a committed manifest entry are never adopted.
            plan.modified += 1
            plan.prepare_hashes.add(digest)
            plan.operations.append(f"~ Recover uncommitted Chroma chunks: {paths[0]}")
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
        config = ingest.get_index_config()
        vectors = ingest.get_embedding_model().encode(
            documents, show_progress_bar=False,
            normalize_embeddings=config.normalize_embeddings
        ).tolist()
        if any(len(vector) != config.embedding_dimension for vector in vectors):
            raise ValueError(f"Embedding model returned dimension {len(vectors[0])}; expected {config.embedding_dimension}")
    return {"ids": ids, "documents": documents, "metadatas": metadatas,
            "embeddings": vectors}


def _write_manifest(data: dict, *, path: Path | None = None, collection_name: str | None = None):
    path = path or manifest_path(collection_name)
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
                 force: bool = False, reindex: bool = False,
                 collection_name: str | None = None) -> SyncPlan:
    with INDEX_LOCK:
        return _sync_library_locked(
            papers_dir, dry_run=dry_run, force=force, reindex=reindex,
            collection_name=collection_name,
        )


def _clear_finalized_pending(collection_name: str | None = None) -> None:
    """Finish a crash-interrupted pending cleanup after metadata became ready."""
    if collection_name:
        return
    metadata = read_metadata(ingest.DB_DIR)
    if metadata is None or metadata.get("status") != "ready":
        return
    if metadata.get("configuration_fingerprint") != ingest.get_index_config().fingerprint:
        return
    path = manifest_path()
    manifest = _read_manifest(path=path)
    pending = manifest.get("pending", {})
    removable = [
        digest for digest, marker in pending.items()
        if digest in manifest.get("documents", {})
        and manifest["documents"][digest].get("status") in {"indexed", "stale", "empty"}
        and manifest["documents"][digest].get("chunk_ids") == marker.get("chunk_ids")
    ]
    if not removable:
        return
    for digest in removable:
        del pending[digest]
    manifest["pending"] = pending
    _write_manifest(manifest, path=path)
    _invalidate_snapshot(path)


def _sync_library_locked(papers_dir: str = "papers", *, dry_run: bool = False,
                         force: bool = False, reindex: bool = False,
                         collection_name: str | None = None) -> SyncPlan:
    if not dry_run:
        _clear_finalized_pending(collection_name)
    plan = plan_sync(papers_dir, dry_run=dry_run, reindex=reindex,
                     collection_name=collection_name)
    if dry_run:
        return plan

    recovering_initial = False
    if not collection_name:
        metadata = read_metadata(ingest.DB_DIR)
        if metadata is None and (plan.stored.get("ids") or []):
            recovering_initial, reason = _initial_recovery_state(
                plan.manifest, plan.stored, metadata
            )
            if not recovering_initial:
                status = (
                    "configuration mismatch" if "configuration" in reason else
                    "incomplete or corrupted" if plan.manifest.get("pending") else
                    "legacy or unversioned"
                )
                raise IndexCompatibilityError(
                    status,
                    "The unversioned Chroma collection cannot be recovered safely: "
                    f"{reason}. Run `python ask.py index rebuild` to replace it.",
                )

    indexed = set(plan.manifest["documents"]) | set(_group_records(plan.stored, plan.files))
    ratio = plan.deleted / len(indexed) if indexed else 0
    threshold = _delete_threshold()
    if plan.deleted and ratio > threshold and not force:
        raise SyncError(
            f"Sync would remove {plan.deleted} of {len(indexed)} indexed documents "
            f"({ratio:.0%}), above SYNC_DELETE_THRESHOLD={threshold:.0%}. "
            "Review the library and rerun with --force to confirm."
        )

    if collection_name:
        import chromadb
        collection = chromadb.PersistentClient(path=ingest.DB_DIR).get_or_create_collection(collection_name)
    else:
        collection = _collection_for_sync()
    if recovering_initial:
        _verify_pending_embeddings(collection, plan.stored)
    active_manifest_path = manifest_path(collection_name)
    previous_committed_revision = _committed_revision(
        plan.manifest.get("documents", {}), ingest.get_index_config().fingerprint
    )
    old_groups = _group_records(plan.stored, plan.files)
    committed_groups = _committed_groups(old_groups, plan.manifest)
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

    # New versions get fresh IDs so a failed replacement cannot overwrite the
    # version that remains committed and visible. A recoverable initial write
    # reuses its manifest-recorded IDs, making retries idempotent.
    for digest, document in list(prepared.items()):
        existing_pending = plan.manifest.get("pending", {}).get(digest, {})
        existing_ids = existing_pending.get("chunk_ids", [])
        if recovering_initial and existing_pending and len(existing_ids) != len(document["ids"]):
            paths = sorted(item["path"] for item in by_hash[digest])
            failed_paths.update(paths)
            plan.failures.append(
                f"{paths[0]}: pending chunk IDs do not match the active chunking result; "
                "rebuild or clear the incomplete index."
            )
            del prepared[digest]
            continue
        if recovering_initial and len(existing_ids) == len(document["ids"]):
            document["ids"] = list(existing_ids)
        else:
            version_id = uuid.uuid4().hex[:16]
            document["ids"] = [f"{chunk_id}-{version_id}" for chunk_id in document["ids"]]

    # Record intended IDs before touching Chroma. A crash during upsert leaves
    # a durable marker so the next run can retry without exposing these IDs.
    pending = dict(plan.manifest.get("pending", {}))
    for digest, document in prepared.items():
        pending[digest] = {
            "paths": sorted(item["path"] for item in by_hash[digest]),
            "chunk_ids": document["ids"], "status": "pending",
            "configuration_fingerprint": ingest.get_index_config().fingerprint,
            "configuration": ingest.get_index_config().canonical(),
        }
    if prepared:
        try:
            _write_manifest({
                "version": 1, "index_version": ingest.get_index_config().fingerprint,
                "index_schema_version": INDEX_SCHEMA_VERSION,
                "index_configuration": ingest.get_index_config().canonical(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "documents": plan.manifest["documents"], "pending": pending,
            }, path=active_manifest_path)
        except OSError as exc:
            raise SyncError(f"Could not persist recovery marker before indexing: {exc}") from exc

    # Only verified writes are candidates for the final manifest commit.
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
                actual_metadata = dict(zip(actual["ids"], actual["metadatas"]))
                expected_metadata = dict(zip(document["ids"], document["metadatas"]))
                if actual_metadata != expected_metadata:
                    raise RuntimeError("Chroma returned metadata that differs from the prepared chunks")
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
            current[digest] = committed_groups.get(digest, [])

    # If a changed PDF failed, retain the previous version for that path and
    # mark it stale. This avoids losing the last successfully indexed content.
    stale_hashes = set()
    for old_hash, entry in plan.manifest["documents"].items():
        if old_hash in current:
            continue
        if set(entry.get("paths", [])) & failed_paths:
            stale_hashes.add(old_hash)
            current[old_hash] = committed_groups.get(old_hash, [])

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
                records = committed_groups.get(digest, records)

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
            "index_version": previous.get("index_version", ingest.get_index_config().fingerprint)
            if digest in stale_hashes else ingest.get_index_config().fingerprint,
            "configuration_fingerprint": ingest.get_index_config().fingerprint,
        }

    pending = {digest: details for digest, details in pending.items()
               if (digest not in current or digest in prepared) and digest in by_hash}
    manifest = {
        "version": 1, "index_version": ingest.get_index_config().fingerprint,
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "index_configuration": ingest.get_index_config().canonical(),
        "updated_at": timestamp, "documents": next_documents, "pending": pending,
    }
    try:
        _write_manifest(manifest, path=active_manifest_path)
    except OSError as exc:
        raise SyncError(f"Index updated but manifest write failed: {exc}") from exc

    committed_revision = _committed_revision(
        next_documents, ingest.get_index_config().fingerprint
    )
    if committed_revision != previous_committed_revision:
        _invalidate_snapshot(active_manifest_path)

    # The manifest is the visibility commit point. Retire old physical chunks
    # afterwards so a failed replacement cannot hide the previous version.
    committed_ids = {
        chunk_id
        for entry in next_documents.values()
        if entry.get("status") in {"indexed", "stale"}
        for chunk_id in entry.get("chunk_ids", [])
    }
    obsolete_by_document = {}
    for digest, records in old_groups.items():
        obsolete = [record for record in records if record["id"] not in committed_ids]
        if obsolete:
            obsolete_by_document[digest] = obsolete

    deleted_successfully = False
    orphan_manifest_changed = False
    for digest, records in obsolete_by_document.items():
        try:
            collection.delete(ids=[row["id"] for row in records])
            deleted_successfully = True
        except Exception as exc:
            plan.failures.append(f"Could not remove obsolete chunks for document {digest[:12]}: {exc}")
            old_entry = plan.manifest["documents"].get(digest, {})
            if digest not in next_documents and old_entry:
                orphan_entry = dict(old_entry)
                orphan_entry["status"] = "orphaned"
                next_documents[digest] = orphan_entry
                orphan_manifest_changed = True

    if orphan_manifest_changed:
        manifest["documents"] = next_documents
        try:
            _write_manifest(manifest, path=active_manifest_path)
        except OSError as exc:
            # The first commit already excludes the obsolete IDs, so failure
            # to record an orphan cannot make those chunks searchable again.
            plan.failures.append(f"Could not record orphaned chunks in manifest: {exc}")

    successfully_prepared = set(prepared) & set(current)
    plan.succeeded = len(successfully_prepared)
    index_changed = bool(successfully_prepared or deleted_successfully or metadata_changed)
    if not collection_name:
        try:
            previous_meta = read_metadata(ingest.DB_DIR)
            if previous_meta is not None:
                check_compatibility(ingest.get_index_config(), previous_meta)
            # `ready` means the manifest and version metadata describe a usable,
            # committed index. Other PDFs may still have recoverable failures.
            should_write = (
                (previous_meta is None and any(
                    digest in next_documents and next_documents[digest].get("chunk_ids")
                    for digest in successfully_prepared
                ))
                or (previous_meta is not None and index_changed)
            )
            if should_write:
                write_metadata(
                    ingest.DB_DIR,
                    new_metadata(ingest.get_index_config(), previous=previous_meta),
                )
        except (OSError, IndexCompatibilityError) as exc:
            raise SyncError(
                f"Index content and manifest were committed, but version metadata could not be safely updated: {exc}"
            ) from exc

    # The manifest's document entries are the visibility commit point. Keep
    # recovery markers until index metadata is finalized, then clear them in a
    # second atomic manifest write. A crash before this write is reconciled by
    # _clear_finalized_pending on the next synchronization.
    finalized = [
        digest for digest, marker in manifest.get("pending", {}).items()
        if digest in manifest["documents"]
        and manifest["documents"][digest].get("status") in {"indexed", "stale", "empty"}
        and manifest["documents"][digest].get("chunk_ids") == marker.get("chunk_ids")
    ]
    if finalized:
        for digest in finalized:
            del manifest["pending"][digest]
        try:
            _write_manifest(manifest, path=active_manifest_path)
        except OSError as exc:
            raise SyncError(
                f"Index is committed and metadata is ready, but recovery markers could not be cleared: {exc}"
            ) from exc

    if index_changed:
        try:
            from src import retrieve
            retrieve._bm25_cache = None
        except ImportError:
            pass
    plan.manifest = manifest
    return plan
