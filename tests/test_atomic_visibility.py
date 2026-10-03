import json
from dataclasses import replace
from pathlib import Path

import pytest

import src.ingest as ingest
import src.retrieve as retrieve
import src.sync as sync
from src.index_config import IndexCompatibilityError, read_metadata


_TOKEN_PREPARE = sync._prepare
_EXPAND_CONTEXT = retrieve.expand_context


@pytest.fixture(autouse=True)
def anchor_visibility_only(monkeypatch):
    # Original visibility tests assert physical anchor IDs, before expansion.
    monkeypatch.setattr(retrieve, "expand_context", lambda question, anchors, corpus, snapshot: anchors)


class AtomicCollection:
    def __init__(self):
        self.rows = {}
        self.fail_after = None
        self.fail_delete = False

    def count(self):
        return len(self.rows)

    def get(self, ids=None, include=None):
        keys = list(self.rows) if ids is None else [key for key in ids if key in self.rows]
        return {
            "ids": keys,
            "documents": [self.rows[key]["text"] for key in keys],
            "metadatas": [dict(self.rows[key]["metadata"]) for key in keys],
        }

    def upsert(self, ids, documents, embeddings, metadatas):
        fail_after = self.fail_after
        self.fail_after = None
        for index, (chunk_id, text, metadata, vector) in enumerate(
            zip(ids, documents, metadatas, embeddings), start=1
        ):
            self.rows[chunk_id] = {
                "id": chunk_id, "text": text, "metadata": dict(metadata),
                "embedding": vector, "distance": 0.01 * index,
            }
            if fail_after == index:
                raise RuntimeError("simulated interrupted write")

    def query(self, query_embeddings, n_results, include):
        rows = sorted(self.rows.values(), key=lambda row: (row["distance"], row["id"]))[:n_results]
        return {
            "ids": [[row["id"] for row in rows]],
            "documents": [[row["text"] for row in rows]],
            "metadatas": [[row["metadata"] for row in rows]],
            "distances": [[row["distance"] for row in rows]],
        }

    def update(self, ids, metadatas):
        for chunk_id, metadata in zip(ids, metadatas):
            self.rows[chunk_id]["metadata"] = dict(metadata)

    def delete(self, ids):
        if self.fail_delete:
            self.fail_delete = False
            raise RuntimeError("simulated deletion failure")
        for chunk_id in ids:
            self.rows.pop(chunk_id, None)


class EmbeddingModel:
    def encode(self, questions, **kwargs):
        class Encoded(list):
            def tolist(self):
                return list(self)

        return Encoded([[1.0] * 384 for _ in questions])


class CrossEncoder:
    def predict(self, pairs, batch_size, show_progress_bar):
        return [float(len(pairs) - index) for index in range(len(pairs))]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    papers = tmp_path / "papers"
    papers.mkdir()
    db = tmp_path / "chroma"
    collection = AtomicCollection()
    monkeypatch.setattr(ingest, "DB_DIR", str(db))
    monkeypatch.setattr(ingest, "get_collection", lambda: collection)
    monkeypatch.setattr(ingest, "open_raw_collection", lambda: collection)
    monkeypatch.setattr(ingest, "validate_collection_ready", lambda current: current)
    monkeypatch.setattr(retrieve, "get_collection", lambda: collection)
    monkeypatch.setattr(retrieve, "get_embedding_model", lambda: EmbeddingModel())
    monkeypatch.setattr(retrieve, "_create_reranker", lambda name: CrossEncoder())
    monkeypatch.setattr(retrieve, "_bm25_cache", None)
    monkeypatch.setattr(sync, "_prepare", lambda item, digest, paths: _prepare(item, digest, paths))
    for name in ("DENSE_CANDIDATES", "BM25_CANDIDATES", "RERANK_CANDIDATES", "FINAL_RESULTS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RERANKER_MIN_SCORE", "0")
    monkeypatch.setenv("MAX_CHUNKS_PER_PAPER", "100")
    monkeypatch.setenv("MMR_LAMBDA", "1")
    monkeypatch.setenv("ADJACENT_CHUNKS", "0")
    return papers, db, collection


def _prepare(item, digest, paths):
    text = item["absolute"].read_text()
    documents = [f"{text} passage {index}" for index in range(3)]
    ids = [f"{digest}-1-{index}" for index in range(3)]
    metadata = [{
        "source": paths[0], "source_paths": json.dumps(paths), "title": Path(paths[0]).stem,
        "section": "Results", "page": index + 1, "document_id": digest,
        "file_hash": digest,
    } for index in range(3)]
    return {
        "ids": ids, "documents": documents, "metadatas": metadata,
        "embeddings": [[1.0] * 384 for _ in documents],
    }


def _add_pdf(papers, name, text):
    path = papers / name
    path.write_text(text)
    return path


def _visible(harness, mode="dense", top_k=10):
    papers, db, collection = harness
    return retrieve.retrieve("passage", top_k=top_k, retrieval_mode=mode)


def test_missing_or_corrupt_manifest_fails_closed(harness):
    papers, db, collection = harness
    collection.rows["unreferenced"] = {
        "id": "unreferenced", "text": "orphan text", "metadata": {},
        "embedding": [1.0] * 384, "distance": 0.01,
    }
    with pytest.raises(IndexCompatibilityError, match="manifest is missing"):
        sync.committed_snapshot(collection_count=collection.count())

    path = sync.manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]")
    with pytest.raises(IndexCompatibilityError, match="manifest"):
        sync.committed_snapshot(collection_count=collection.count())


def test_committed_snapshot_cache_checks_active_fingerprint(harness, monkeypatch):
    papers, db, collection = harness
    _add_pdf(papers, "paper.pdf", "committed")
    sync.sync_library(str(papers))
    assert sync.committed_snapshot(collection_count=collection.count()).chunk_ids

    config = ingest.get_index_config()
    monkeypatch.setattr(ingest, "get_index_config", lambda: replace(config, chunk_size=config.chunk_size + 1))
    with pytest.raises(IndexCompatibilityError, match="fingerprint"):
        sync.committed_snapshot(collection_count=collection.count())


@pytest.mark.parametrize("mode", retrieve.RETRIEVAL_MODES)
def test_interrupted_initial_write_is_invisible_in_every_mode(harness, mode):
    papers, db, collection = harness
    _add_pdf(papers, "paper.pdf", "first")
    collection.fail_after = 1

    result = sync.sync_library(str(papers))

    assert result.failures
    assert collection.count() == 1
    manifest = json.loads(sync.manifest_path().read_text())
    assert manifest["documents"] == {}
    assert len(manifest["pending"]) == 1
    assert read_metadata(db) is None
    assert _visible(harness, mode) == []


def test_recovery_commits_all_chunks_then_clears_pending_and_invalidates_bm25(harness):
    papers, db, collection = harness
    _add_pdf(papers, "paper.pdf", "recover")
    collection.fail_after = 1
    sync.sync_library(str(papers))
    pending_ids = next(iter(json.loads(sync.manifest_path().read_text())["pending"].values()))["chunk_ids"]
    assert len(pending_ids) == 3 and collection.count() == 1
    assert _visible(harness, "hybrid") == []

    result = sync.sync_library(str(papers))

    manifest = json.loads(sync.manifest_path().read_text())
    committed_ids = manifest["documents"][next(iter(manifest["documents"]))]["chunk_ids"]
    assert result.succeeded == 1
    assert manifest["pending"] == {}
    assert committed_ids == pending_ids
    assert set(collection.rows) == set(committed_ids)
    for mode in retrieve.RETRIEVAL_MODES:
        hits = _visible(harness, mode)
        assert len(hits) == 3
        assert {hit["id"] for hit in hits} == set(committed_ids)
        assert all(hit["metadata"]["document_id"] == next(iter(manifest["documents"])) for hit in hits)


def test_failed_replacement_keeps_old_version_visible_until_new_commit(harness):
    papers, db, collection = harness
    path = _add_pdf(papers, "paper.pdf", "version one")
    sync.sync_library(str(papers))
    old_ids = set(next(iter(json.loads(sync.manifest_path().read_text())["documents"].values()))["chunk_ids"])
    old_hash = next(iter(json.loads(sync.manifest_path().read_text())["documents"]))

    path.write_text("version two")
    collection.fail_after = 1
    failed = sync.sync_library(str(papers))
    pending = json.loads(sync.manifest_path().read_text())["pending"]
    pending_ids = set(next(iter(pending.values()))["chunk_ids"])
    assert failed.failures
    assert old_ids <= set(collection.rows)
    assert old_hash in json.loads(sync.manifest_path().read_text())["documents"]
    for mode in retrieve.RETRIEVAL_MODES:
        assert {hit["id"] for hit in _visible(harness, mode)} == old_ids
    assert pending_ids.isdisjoint({hit["id"] for hit in _visible(harness, "dense")})

    succeeded = sync.sync_library(str(papers))
    manifest = json.loads(sync.manifest_path().read_text())
    new_hash = ingest.file_hash(path)
    new_ids = set(manifest["documents"][new_hash]["chunk_ids"])
    assert succeeded.succeeded == 1
    assert old_hash not in manifest["documents"]
    assert new_ids <= set(collection.rows)
    assert old_ids.isdisjoint(set(collection.rows))
    assert {hit["id"] for hit in _visible(harness, "hybrid-rerank")} <= new_ids


def test_final_manifest_write_failure_leaves_new_ids_invisible_and_retryable(harness, monkeypatch):
    papers, db, collection = harness
    _add_pdf(papers, "paper.pdf", "commit failure")
    original_write = sync._write_manifest
    calls = 0

    def fail_commit(data, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated final manifest failure")
        return original_write(data, **kwargs)

    monkeypatch.setattr(sync, "_write_manifest", fail_commit)
    with pytest.raises(sync.SyncError, match="manifest write failed"):
        sync.sync_library(str(papers))
    pending_manifest = json.loads(sync.manifest_path().read_text())
    assert len(pending_manifest["pending"]) == 1
    assert pending_manifest["documents"] == {}
    assert _visible(harness, "dense") == []

    monkeypatch.setattr(sync, "_write_manifest", original_write)
    sync.sync_library(str(papers))
    assert len(_visible(harness, "dense")) == 3


def test_dense_and_hybrid_overfetch_past_pending_top_positions(harness, monkeypatch):
    papers, db, collection = harness
    committed_rows = [
        {"id": f"committed-{i}", "text": f"matching passage {i}",
         "metadata": {"source": "good.pdf", "title": "Good", "page": i + 1,
                      "document_id": "good-hash"}, "distance": 0.5 + i / 10}
        for i in range(3)
    ]
    pending_rows = [
        {"id": f"pending-{i}", "text": "matching passage pending", "metadata": {
            "source": "bad.pdf", "title": "Bad", "page": 1, "document_id": "bad-hash",
        }, "distance": 0.01 + i / 100}
        for i in range(5)
    ]
    collection.rows = {
        row["id"]: {**row, "embedding": [1.0] * 384}
        for row in committed_rows + pending_rows
    }
    snapshot = sync.CommittedSnapshot(
        "committed-only", frozenset(row["id"] for row in committed_rows),
        {row["id"]: {"document_id": "good-hash", "paths": ["good.pdf"]}
         for row in committed_rows},
    )
    monkeypatch.setattr(retrieve, "_get_committed_snapshot", lambda current: snapshot)
    monkeypatch.setenv("DENSE_CANDIDATES", "2")
    monkeypatch.setenv("BM25_CANDIDATES", "2")

    dense = _visible(harness, "dense", 2)
    hybrid = _visible(harness, "hybrid", 2)

    assert len(dense) == len(hybrid) == 2
    assert all(hit["id"].startswith("committed-") for hit in dense + hybrid)


def test_bm25_cache_tracks_committed_revision_not_pending_write(harness):
    papers, db, collection = harness
    path = _add_pdf(papers, "paper.pdf", "stable document")
    sync.sync_library(str(papers))
    old_hits = retrieve._bm25_search("stable", collection, 5, sync.committed_snapshot())
    cache_before = retrieve._bm25_cache
    assert old_hits and cache_before is not None

    path.write_text("replacement document")
    collection.fail_after = 1
    sync.sync_library(str(papers))
    old_snapshot = sync.committed_snapshot(collection_count=collection.count())
    current_hits = retrieve._bm25_search("stable", collection, 5, old_snapshot)
    assert retrieve._bm25_cache is cache_before
    assert {hit["id"] for hit in current_hits} == {hit["id"] for hit in old_hits}

    sync.sync_library(str(papers))
    assert retrieve._bm25_cache is None
    new_hits = retrieve._bm25_search("replacement", collection, 5, sync.committed_snapshot())
    assert new_hits and all("replacement" in hit["text"] for hit in new_hits)


def test_failed_obsolete_delete_keeps_old_chunks_physical_but_invisible(harness):
    papers, db, collection = harness
    path = _add_pdf(papers, "paper.pdf", "old searchable words")
    sync.sync_library(str(papers))
    old_ids = set(next(iter(json.loads(sync.manifest_path().read_text())["documents"].values()))["chunk_ids"])
    path.write_text("new searchable words")
    collection.fail_delete = True

    result = sync.sync_library(str(papers))

    manifest = json.loads(sync.manifest_path().read_text())
    new_ids = set(manifest["documents"][ingest.file_hash(path)]["chunk_ids"])
    assert result.failures
    assert old_ids <= set(collection.rows)
    assert old_ids.isdisjoint({hit["id"] for hit in _visible(harness, "dense")})
    assert new_ids <= {hit["id"] for hit in _visible(harness, "hybrid-rerank")}


def test_real_chroma_partial_insert_is_invisible_then_recovered(tmp_path, monkeypatch):
    chromadb = pytest.importorskip("chromadb")
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "paper.pdf").write_text("chroma integration")
    db = tmp_path / "chroma"
    monkeypatch.setattr(ingest, "DB_DIR", str(db))
    client = chromadb.PersistentClient(path=str(db))
    raw = client.get_or_create_collection("papers")

    class InterruptOnce:
        fail = True

        def __getattr__(self, name):
            return getattr(raw, name)

        def upsert(self, ids, documents, embeddings, metadatas):
            if self.fail:
                self.fail = False
                raw.upsert(ids=ids[:1], documents=documents[:1],
                           embeddings=embeddings[:1], metadatas=metadatas[:1])
                raise RuntimeError("simulated crash after first Chroma row")
            raw.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    proxy = InterruptOnce()
    monkeypatch.setattr(ingest, "get_collection", lambda: proxy)
    monkeypatch.setattr(ingest, "open_raw_collection", lambda: proxy)
    monkeypatch.setattr(retrieve, "get_collection", lambda: raw)
    monkeypatch.setattr(retrieve, "get_embedding_model", lambda: EmbeddingModel())
    monkeypatch.setattr(sync, "_prepare", lambda item, digest, paths: _prepare(item, digest, paths))

    failed = sync.sync_library(str(papers))
    assert failed.failures and raw.count() == 1
    assert retrieve.retrieve("chroma", retrieval_mode="dense") == []

    recovered = sync.sync_library(str(papers))
    assert recovered.succeeded == 1
    hits = retrieve.retrieve("chroma", top_k=10, retrieval_mode="hybrid")
    assert len(hits) == 3
    assert all(hit["source"] == "paper.pdf" for hit in hits)
    assert all(hit["page"] in {1, 2, 3} for hit in hits)


@pytest.mark.parametrize("initial", [True, False])
def test_stored_context_survives_interrupted_write_and_version_recovery(harness, monkeypatch, initial):
    from passage_helpers import CharacterTokenizer
    monkeypatch.setattr(retrieve, "expand_context", _EXPAND_CONTEXT)
    papers, db, collection = harness
    config = replace(ingest.get_index_config(), chunk_size=40, chunk_overlap=10)
    monkeypatch.setattr(ingest, "get_index_config", lambda: config)
    monkeypatch.setattr(ingest, "tokenizer_limits", lambda: (CharacterTokenizer(), 510))
    monkeypatch.setattr(ingest, "get_embedding_model", lambda: EmbeddingModel())
    monkeypatch.setattr(ingest, "extract_pages", lambda path: [(1, path.read_text()), (2, "Continuation.")])
    monkeypatch.setattr(sync, "_prepare", _TOKEN_PREPARE)
    old_text = "# Methods\n\nOld anchor.\n\nOld neighbor."
    path = _add_pdf(papers, "paper.pdf", old_text)
    if not initial:
        assert not sync.sync_library(str(papers)).failures
    new_text = "# Methods\n\nNew anchor.\n\nNew neighbor."
    path.write_text(new_text)
    collection.fail_after = 1
    assert sync.sync_library(str(papers)).failures
    visible = _visible(harness, top_k=10)
    if initial:
        assert visible == []
    else:
        assert any("Old anchor." in hit["text"] for hit in visible)
        assert all("New" not in hit["text"] for hit in visible)
    assert not sync.sync_library(str(papers)).failures
    visible = _visible(harness, top_k=10)
    assert any("New anchor." in hit["text"] for hit in visible)
    assert all("Old" not in hit["text"] for hit in visible)
    assert len({hit["metadata"]["document_version"] for hit in visible}) == 1
    assert all(hit["page"] in {1, 2} for hit in visible)
