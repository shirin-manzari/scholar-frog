"""Typed, validated configuration and metadata for the persistent vector index."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

INDEX_SCHEMA_VERSION = 1
CHUNKING_VERSION = "markdown-heading-paragraph-v1"
TEXT_EXTRACTION_VERSION = "pymupdf4llm-fallback-v1"
CONFIG_FILE = "scholarq.toml"


@dataclass(frozen=True)
class IndexConfig:
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_revision: str | None = None
    embedding_dimension: int = 384
    normalize_embeddings: bool = True
    chunk_size: int = 800
    chunk_overlap: int = 150
    chunking_strategy: str = "markdown-heading-paragraph"
    chunking_version: str = CHUNKING_VERSION
    text_extraction_version: str = TEXT_EXTRACTION_VERSION

    def __post_init__(self):
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        if self.embedding_revision is not None and not self.embedding_revision.strip():
            raise ValueError("embedding_revision must be omitted or a non-empty revision")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size - 1")
        for name in ("chunking_strategy", "chunking_version", "text_extraction_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.chunking_strategy != "markdown-heading-paragraph":
            raise ValueError("chunking_strategy must be 'markdown-heading-paragraph'")

    def canonical(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(raw.encode()).hexdigest()


_ENV = {
    "embedding_model": ("SCHOLARQ_EMBEDDING_MODEL", str),
    "embedding_revision": ("SCHOLARQ_EMBEDDING_REVISION", str),
    "embedding_dimension": ("SCHOLARQ_EMBEDDING_DIMENSION", int),
    "normalize_embeddings": ("SCHOLARQ_NORMALIZE_EMBEDDINGS", lambda v: _bool(v)),
    "chunk_size": ("SCHOLARQ_CHUNK_SIZE", int),
    "chunk_overlap": ("SCHOLARQ_CHUNK_OVERLAP", int),
    "chunking_strategy": ("SCHOLARQ_CHUNKING_STRATEGY", str),
}
_USER_OPTIONS = frozenset({
    "embedding_model", "embedding_revision", "embedding_dimension",
    "normalize_embeddings", "chunk_size", "chunk_overlap", "chunking_strategy",
})


def _bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError("expected true/false")
    return lowered in {"true", "1", "yes"}


def load_index_config(*, overrides: dict | None = None, config_path: str | Path = CONFIG_FILE) -> IndexConfig:
    values = IndexConfig().canonical()
    path = Path(config_path)
    if path.exists():
        try:
            try:
                import tomllib
            except ImportError:  # Python 3.10
                import tomli as tomllib
            with path.open("rb") as stream:
                data = tomllib.load(stream)
            section = data.get("index", data)
            unknown = set(section) - _USER_OPTIONS
            if unknown:
                raise ValueError(f"Unknown index configuration option(s): {', '.join(sorted(unknown))}")
            values.update(section)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid configuration in {path}: {exc}") from exc
    for key, (env_name, cast) in _ENV.items():
        if env_name in os.environ:
            try:
                values[key] = cast(os.environ[env_name])
            except ValueError as exc:
                raise ValueError(f"{env_name} has an invalid value: {exc}") from exc
    for key, value in (overrides or {}).items():
        if value is not None:
            if key not in _USER_OPTIONS:
                raise ValueError(f"Unknown index configuration option: {key}")
            values[key] = value
    return IndexConfig(**values)


def metadata_path(db_dir: str | Path) -> Path:
    return Path(db_dir) / "index_metadata.json"


def read_metadata(db_dir: str | Path) -> dict | None:
    path = metadata_path(db_dir)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("metadata root must be an object")
        return value
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise IndexCompatibilityError("incomplete or corrupted", f"Cannot read index metadata {path}: {exc}") from exc


def write_metadata(db_dir: str | Path, metadata: dict) -> None:
    path = metadata_path(db_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        if temporary:
            temporary.unlink(missing_ok=True)
        raise


class IndexCompatibilityError(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def check_compatibility(active: IndexConfig, metadata: dict | None, *, collection_count: int | None = None) -> str:
    if metadata is None:
        if collection_count:
            raise IndexCompatibilityError("legacy or unversioned", "The existing Chroma collection has no version metadata (legacy or unversioned). Run `python ask.py index rebuild` to create a versioned index.")
        return "missing"
    if metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise IndexCompatibilityError("unsupported schema version", f"Index schema {metadata.get('schema_version')!r} is unsupported (supported: {INDEX_SCHEMA_VERSION}).")
    if metadata.get("status") != "ready" or not isinstance(metadata.get("configuration"), dict):
        raise IndexCompatibilityError("incomplete or corrupted", "Index metadata is incomplete or the index is not marked ready.")
    if metadata.get("configuration_fingerprint") != active.fingerprint:
        stored = metadata.get("configuration", {})
        changes = [f"{key}: stored={stored.get(key)!r}, active={value!r}" for key, value in active.canonical().items() if stored.get(key) != value]
        raise IndexCompatibilityError("configuration mismatch", "Index configuration differs; rebuild required. " + "; ".join(changes))
    return "compatible"


def new_metadata(config: IndexConfig, *, status: str = "ready", previous: dict | None = None) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "configuration": config.canonical(),
        "resolved_embedding_revision": config.embedding_revision,
        "embedding_dimension": config.embedding_dimension,
        "configuration_fingerprint": config.fingerprint,
        "created_at": (previous or {}).get("created_at", now),
        "last_successful_update_at": now,
        "status": status,
        "active_collection": (previous or {}).get("active_collection", "papers"),
        "manifest_name": (previous or {}).get("manifest_name", "documents.json"),
    }
