import json
from pathlib import Path

import pytest

import src.ingest as ingest
import src.retrieve as retrieve
import src.sync as sync
from src.index_config import check_compatibility, load_index_config, read_metadata


class MemoryCollection:
    def __init__(self):
        self.rows = {}
        self.fail_upsert_once = False
        self.fail_delete_once = False

    def get(self, ids=None, include=None):
        wanted = list(self.rows) if ids is None else [key for key in ids if key in self.rows]
        return {
            "ids": wanted,
            "documents": [self.rows[key]["text"] for key in wanted],
            "metadatas": [dict(self.rows[key]["metadata"]) for key in wanted],
        }

    def upsert(self, ids, documents, embeddings, metadatas):
        for key, text, metadata in zip(ids, documents, metadatas):
            self.rows[key] = {"text": text, "metadata": dict(metadata)}
            if self.fail_upsert_once:
                self.fail_upsert_once = False
                raise RuntimeError("mock interrupted insert")

    def update(self, ids, metadatas):
        for key, metadata in zip(ids, metadatas):
            self.rows[key]["metadata"] = dict(metadata)

    def delete(self, ids):
        if self.fail_delete_once:
            self.fail_delete_once = False
            raise RuntimeError("mock delete failure")
        for key in ids:
            self.rows.pop(key, None)


@pytest.fixture
def library(tmp_path, monkeypatch):
    root = tmp_path / "papers"
    root.mkdir()
    db = tmp_path / "db"
    collection = MemoryCollection()
    monkeypatch.setattr(ingest, "DB_DIR", str(db))
    monkeypatch.setattr(ingest, "get_collection", lambda: collection)
    prepared = []

    def fake_prepare(item, digest, paths):
        prepared.append((digest, tuple(paths)))
        text = item["absolute"].read_text()
        if text.startswith("FAIL"):
            raise ValueError("mock extraction failed")
        if text == "EMPTY":
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        chunk_id = f"{digest}-1-0"
        metadata = {
            "source": paths[0], "source_paths": json.dumps(paths), "title": Path(paths[0]).stem,
            "section": "Results", "page": 1, "document_id": digest, "file_hash": digest,
        }
        return {"ids": [chunk_id], "documents": [text], "metadatas": [metadata], "embeddings": [[0.1]]}

    monkeypatch.setattr(sync, "_prepare", fake_prepare)
    return root, db, collection, prepared


def add_pdf(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_initial_index_and_unchanged_sync_do_not_reembed(library):
    root, _, collection, prepared = library
    add_pdf(root, "a.pdf", "paper content")
    first = sync.sync_library(str(root))
    assert first.added == 1
    assert len(collection.rows) == 1
    assert len(prepared) == 1

    second = sync.sync_library(str(root))
    assert second.unchanged == 1
    assert len(prepared) == 1
    assert len(collection.rows) == 1


def test_add_modify_move_and_delete_sync(library):
    root, _, collection, prepared = library
    path = add_pdf(root, "a.pdf", "v1")
    sync.sync_library(str(root))
    path.write_text("v2")
    changed = sync.sync_library(str(root))
    assert changed.modified == 1
    assert len(collection.rows) == 1
    assert next(iter(collection.rows.values()))["text"] == "v2"

    (root / "archive").mkdir()
    stable_chunk = next(iter(collection.rows))
    path.rename(root / "archive" / "a.pdf")
    moved = sync.sync_library(str(root))
    assert moved.renamed == 1
    assert len(prepared) == 2
    assert list(collection.rows) == [stable_chunk]
    assert next(iter(collection.rows.values()))["metadata"]["source"] == "archive/a.pdf"

    (root / "archive" / "a.pdf").unlink()
    deleted = sync.sync_library(str(root), force=True)
    assert deleted.deleted == 1
    assert not collection.rows


def test_duplicate_copies_share_chunks_and_primary_source_updates(library):
    root, _, collection, prepared = library
    add_pdf(root, "z.pdf", "same bytes")
    sync.sync_library(str(root))
    add_pdf(root, "a.pdf", "same bytes")
    duplicate = sync.sync_library(str(root))
    assert duplicate.duplicated == 1
    assert len(prepared) == 1
    assert len(collection.rows) == 1
    row = next(iter(collection.rows.values()))
    assert row["metadata"]["source"] == "a.pdf"
    assert json.loads(row["metadata"]["source_paths"]) == ["a.pdf", "z.pdf"]

    (root / "a.pdf").unlink()
    sync.sync_library(str(root))
    assert len(collection.rows) == 1
    assert next(iter(collection.rows.values()))["metadata"]["source"] == "z.pdf"

    (root / "z.pdf").unlink()
    with pytest.raises(sync.SyncError, match="--force"):
        sync.sync_library(str(root))
    sync.sync_library(str(root), force=True)
    assert not collection.rows


def test_failed_replacement_keeps_previous_index_as_stale(library):
    root, db, collection, prepared = library
    path = add_pdf(root, "a.pdf", "v1")
    sync.sync_library(str(root))
    old_id = next(iter(collection.rows))
    path.write_text("FAIL")
    result = sync.sync_library(str(root))
    assert result.failures
    assert list(collection.rows) == [old_id]
    manifest = json.loads((db / sync.manifest_path().name).read_text())
    assert any(document["status"] == "stale" for document in manifest["documents"].values())
    assert len(prepared) == 2


def test_failed_new_document_does_not_stop_other_additions(library):
    root, db, collection, prepared = library
    add_pdf(root, "bad.pdf", "FAIL")
    add_pdf(root, "good.pdf", "ok")
    result = sync.sync_library(str(root))
    assert len(result.failures) == 1
    assert result.succeeded == 1
    assert len(collection.rows) == 1
    assert next(iter(collection.rows.values()))["metadata"]["source"] == "good.pdf"
    manifest = json.loads(sync.manifest_path().read_text())
    assert len(manifest["documents"]) == 1
    metadata = read_metadata(db)
    assert metadata["status"] == "ready"
    assert check_compatibility(load_index_config(), metadata, collection_count=1) == "compatible"
    assert len(prepared) == 2


def test_sync_cli_reports_partial_success_and_returns_nonzero(library, capsys):
    import ask

    root, _, _, _ = library
    add_pdf(root, "bad.pdf", "FAIL")
    add_pdf(root, "good.pdf", "ok")

    exit_code = ask.sync_main(["--papers", str(root)])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Successfully indexed" in output and "1" in output
    assert "Failed" in output
    assert "mock extraction failed" in output


def test_partial_ingestion_retries_only_the_failed_pdf(library):
    root, db, collection, prepared = library
    add_pdf(root, "bad.pdf", "FAIL")
    add_pdf(root, "good.pdf", "already indexed")
    first = sync.sync_library(str(root))
    assert first.failures and first.succeeded == 1
    first_metadata = read_metadata(db)
    assert len(prepared) == 2

    (root / "bad.pdf").write_text("fixed")
    second = sync.sync_library(str(root))
    assert not second.failures and second.succeeded == 1
    assert len(prepared) == 3
    assert len(collection.rows) == 2
    assert set(read_metadata(db)["configuration"].items()) == set(first_metadata["configuration"].items())
    assert len(second.manifest["documents"]) == 2


def test_all_initial_documents_failing_does_not_create_ready_metadata(library):
    root, db, collection, _ = library
    add_pdf(root, "bad-a.pdf", "FAIL A")
    add_pdf(root, "bad-b.pdf", "FAIL B")

    result = sync.sync_library(str(root))

    assert len(result.failures) == 2
    assert result.succeeded == 0
    assert not collection.rows
    assert not (db / "index_metadata.json").exists()
    assert read_metadata(db) is None
    assert json.loads(sync.manifest_path().read_text())["documents"] == {}


def test_existing_ready_index_remains_compatible_after_partial_sync(library):
    root, db, collection, _ = library
    add_pdf(root, "existing.pdf", "existing")
    sync.sync_library(str(root))
    previous = read_metadata(db)
    add_pdf(root, "bad.pdf", "FAIL")
    add_pdf(root, "new.pdf", "new evidence")

    result = sync.sync_library(str(root))

    assert result.failures and result.succeeded == 1
    assert len(collection.rows) == 2
    current = read_metadata(db)
    assert current["active_collection"] == previous["active_collection"]
    assert check_compatibility(load_index_config(), current, collection_count=2) == "compatible"
    assert current["last_successful_update_at"] != previous["last_successful_update_at"]
    manifest = json.loads(sync.manifest_path().read_text())
    assert len(manifest["documents"]) == 2


def test_all_failed_new_documents_do_not_rewrite_existing_metadata(library):
    root, db, _, _ = library
    add_pdf(root, "existing.pdf", "existing")
    sync.sync_library(str(root))
    before = (db / "index_metadata.json").read_bytes()
    add_pdf(root, "bad.pdf", "FAIL")

    result = sync.sync_library(str(root))

    assert result.failures and result.succeeded == 0
    assert (db / "index_metadata.json").read_bytes() == before


def test_dry_run_has_no_manifest_or_index_mutations(library):
    root, db, collection, prepared = library
    add_pdf(root, "new.pdf", "content")
    result = sync.sync_library(str(root), dry_run=True)
    assert result.added == 1
    assert not db.exists()
    assert not collection.rows
    assert prepared == []
    assert result.operations == ["+ Index new.pdf"]


def test_missing_library_and_deletion_safeguard_are_safe(library):
    root, _, collection, _ = library
    add_pdf(root, "a.pdf", "content")
    sync.sync_library(str(root))
    with pytest.raises(sync.SyncError, match="does not exist"):
        sync.sync_library(str(root / "missing"), force=True)
    (root / "a.pdf").unlink()
    with pytest.raises(sync.SyncError, match="SYNC_DELETE_THRESHOLD"):
        sync.sync_library(str(root))
    assert collection.rows


def test_empty_pdfs_are_manifested_and_not_reprocessed(library):
    root, db, _, prepared = library
    add_pdf(root, "empty.pdf", "EMPTY")
    first = sync.sync_library(str(root))
    assert first.added == 1
    second = sync.sync_library(str(root))
    assert second.unchanged == 1
    assert len(prepared) == 1
    manifest = json.loads((db / sync.manifest_path().name).read_text())
    item = next(iter(manifest["documents"].values()))
    assert item["status"] == "empty" and item["chunk_count"] == 0


def test_manifest_failure_before_index_write_is_safe(library, monkeypatch):
    root, db, collection, prepared = library
    add_pdf(root, "a.pdf", "recover me")
    original_write = sync._write_manifest
    monkeypatch.setattr(sync, "_write_manifest", lambda data, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(sync.SyncError, match="recovery marker"):
        sync.sync_library(str(root))
    assert not collection.rows
    monkeypatch.setattr(sync, "_write_manifest", original_write)
    recovered = sync.sync_library(str(root))
    assert recovered.added == 1
    assert len(prepared) == 2


def test_final_manifest_failure_does_not_mark_new_index_ready(library, monkeypatch):
    root, db, collection, _ = library
    add_pdf(root, "a.pdf", "content")
    original_write = sync._write_manifest
    calls = 0

    def fail_final_write(data, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        return original_write(data, **kwargs)

    monkeypatch.setattr(sync, "_write_manifest", fail_final_write)
    with pytest.raises(sync.SyncError, match="manifest write failed"):
        sync.sync_library(str(root))
    assert collection.rows
    assert read_metadata(db) is None


def test_partial_chroma_insert_is_retried_from_pending_marker(library):
    root, db, collection, prepared = library
    add_pdf(root, "a.pdf", "recover partial insert")
    collection.fail_upsert_once = True
    failed = sync.sync_library(str(root))
    assert failed.failures
    assert collection.rows
    pending = json.loads((db / sync.manifest_path().name).read_text())["pending"]
    assert pending

    recovered = sync.sync_library(str(root))
    assert not recovered.failures
    assert len(prepared) == 2
    manifest = json.loads((db / sync.manifest_path().name).read_text())
    assert manifest["pending"] == {}
    assert len(manifest["documents"]) == 1


def test_delete_failure_is_recorded_and_retried(library):
    root, _, collection, _ = library
    path = add_pdf(root, "a.pdf", "content")
    sync.sync_library(str(root))
    path.unlink()
    collection.fail_delete_once = True

    result = sync.sync_library(str(root), force=True)
    assert result.failures
    assert collection.rows
    assert next(iter(result.manifest["documents"].values()))["status"] == "orphaned"

    sync.sync_library(str(root), force=True)
    assert not collection.rows


def test_bm25_cache_invalidated_after_successful_index_change(library):
    root, _, _, _ = library
    add_pdf(root, "a.pdf", "text")
    retrieve._bm25_cache = ("sentinel",)
    sync.sync_library(str(root))
    assert retrieve._bm25_cache is None


def test_real_chroma_collection_supports_sync_operations(tmp_path, monkeypatch):
    chromadb = pytest.importorskip("chromadb")
    root = tmp_path / "papers"
    root.mkdir()
    db = tmp_path / "chroma"
    monkeypatch.setattr(ingest, "DB_DIR", str(db))
    client = chromadb.PersistentClient(path=str(db))
    collection = client.get_or_create_collection("sync-test")
    monkeypatch.setattr(ingest, "get_collection", lambda: collection)

    def prepare(item, digest, paths):
        chunk_id = f"{digest}-1-0"
        metadata = {"source": paths[0], "source_paths": json.dumps(paths),
                    "title": paths[0], "section": "Results", "page": 1,
                    "document_id": digest, "file_hash": digest}
        return {"ids": [chunk_id], "documents": ["indexed passage"],
                "metadatas": [metadata], "embeddings": [[1.0, 0.0]]}

    monkeypatch.setattr(sync, "_prepare", prepare)
    add_pdf(root, "nested/one.pdf", "identical")
    sync.sync_library(str(root))
    assert collection.count() == 1
    manifest_before = sync.manifest_path().read_bytes()
    add_pdf(root, "next.pdf", "new content")
    preview = sync.sync_library(str(root), dry_run=True)
    assert preview.added == 1
    assert collection.count() == 1
    assert sync.manifest_path().read_bytes() == manifest_before
    (root / "next.pdf").unlink()

    add_pdf(root, "copy.pdf", "identical")
    sync.sync_library(str(root))
    assert collection.count() == 1
    row = collection.get(include=["metadatas"])
    assert row["metadatas"][0]["source"] == "copy.pdf"

    (root / "copy.pdf").unlink()
    sync.sync_library(str(root))
    assert collection.count() == 1
    (root / "nested/one.pdf").unlink()
    sync.sync_library(str(root), force=True)
    assert collection.count() == 0


def test_uncommitted_hash_only_record_is_reindexed_instead_of_adopted(library):
    root, _, collection, prepared = library
    path = add_pdf(root, "old-name.pdf", "legacy contents")
    digest = ingest.file_hash(path)
    legacy_id = f"{digest[:16]}-1-0"
    collection.rows[legacy_id] = {
        "text": "existing passage",
        "metadata": {
            "source": "before-rename.pdf", "title": "Legacy", "page": 1,
            "file_hash": digest[:16],
        },
    }
    path.rename(root / "new-name.pdf")

    result = sync.sync_library(str(root))
    assert result.succeeded == 1
    assert len(prepared) == 1
    assert legacy_id not in collection.rows
    assert len(collection.rows) == 1
    row = next(iter(collection.rows.values()))
    assert row["metadata"]["source"] == "new-name.pdf"
