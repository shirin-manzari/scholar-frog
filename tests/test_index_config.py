import json
from pathlib import Path

import pytest

from src.index_config import (
    IndexCompatibilityError, IndexConfig, check_compatibility,
    load_index_config, new_metadata, read_metadata, write_metadata,
)


def test_configuration_defaults_validation_and_env_precedence(tmp_path, monkeypatch):
    path = tmp_path / "scholarq.toml"
    path.write_text('[index]\nchunk_size = 900\nchunk_overlap = 120\n')
    monkeypatch.setenv("SCHOLARQ_CHUNK_SIZE", "1000")
    config = load_index_config(config_path=path, overrides={"chunk_overlap": 99})
    assert (config.chunk_size, config.chunk_overlap) == (1000, 99)
    with pytest.raises(ValueError, match="chunk_overlap"):
        IndexConfig(chunk_size=100, chunk_overlap=100)


def test_fingerprints_are_canonical_and_only_index_settings_affect_them():
    first = IndexConfig()
    equivalent = IndexConfig(**json.loads(json.dumps(first.canonical())))
    assert first.fingerprint == equivalent.fingerprint
    assert IndexConfig(embedding_model="other").fingerprint != first.fingerprint
    assert IndexConfig(embedding_revision="abc").fingerprint != first.fingerprint
    assert IndexConfig(chunk_size=801).fingerprint != first.fingerprint
    assert IndexConfig(chunk_overlap=149).fingerprint != first.fingerprint
    assert IndexConfig(chunking_version="new").fingerprint != first.fingerprint
    assert IndexConfig(text_extraction_version="new").fingerprint != first.fingerprint


def test_metadata_atomic_roundtrip_and_compatibility_states(tmp_path):
    config = IndexConfig()
    assert check_compatibility(config, None) == "missing"
    with pytest.raises(IndexCompatibilityError, match="legacy"):
        check_compatibility(config, None, collection_count=2)
    metadata = new_metadata(config)
    write_metadata(tmp_path, metadata)
    assert read_metadata(tmp_path) == metadata
    assert check_compatibility(config, metadata) == "compatible"
    with pytest.raises(IndexCompatibilityError, match="configuration differs"):
        check_compatibility(IndexConfig(chunk_size=700), metadata)
    with pytest.raises(IndexCompatibilityError, match="unsupported"):
        check_compatibility(config, {**metadata, "schema_version": 900})


def test_corrupt_metadata_is_reported(tmp_path):
    (tmp_path / "index_metadata.json").write_text("{")
    with pytest.raises(IndexCompatibilityError, match="Cannot read"):
        read_metadata(tmp_path)


def test_rebuild_activates_validated_staging_collection_and_preserves_old_on_failure(tmp_path, monkeypatch):
    chromadb = pytest.importorskip("chromadb")
    import ask
    from src import ingest, sync

    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "paper.pdf").write_text("mock source")
    db = tmp_path / "chroma"
    monkeypatch.setattr(ask, "DB_DIR", str(db))
    monkeypatch.setattr(ingest, "DB_DIR", str(db))
    monkeypatch.setattr(sync, "_prepare", lambda item, digest, paths: {
        "ids": [f"{digest}-1-0"], "documents": ["evidence"],
        "metadatas": [{"source": paths[0], "source_paths": json.dumps(paths),
                       "title": "Paper", "section": "Results", "page": 1,
                       "document_id": digest, "file_hash": digest}],
        "embeddings": [[0.1] * 384],
    })
    assert ask.index_main(["rebuild", "--papers", str(papers)]) == 0
    stored = read_metadata(db)
    assert stored["active_collection"].startswith("papers_staging_")
    client = chromadb.PersistentClient(path=str(db))
    old_name = stored["active_collection"]
    old_count = client.get_collection(old_name).count()
    manifest = (db / stored["manifest_name"]).read_bytes()

    monkeypatch.setattr(sync, "_prepare", lambda *args: (_ for _ in ()).throw(ValueError("stop")))
    assert ask.index_main(["rebuild", "--papers", str(papers)]) == 1
    assert read_metadata(db)["active_collection"] == old_name
    assert client.get_collection(old_name).count() == old_count
    assert (db / stored["manifest_name"]).read_bytes() == manifest
