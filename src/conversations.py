"""Local SQLite storage for web conversations.

The schema keeps conversation scope and message metadata separate from visible
content so a future migration does not depend on the browser's rendering.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class ConversationError(RuntimeError):
    pass


class ConversationNotFound(ConversationError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_title(question: str) -> str:
    words = question.strip().rstrip("?.! ").split()
    title = " ".join(words[:7]) or "New chat"
    return title + ("…" if len(words) > 7 else "")


class ConversationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def _connect(self):
        connection = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=10)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 10000")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    scope_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    resolved_query TEXT,
                    scope_json TEXT NOT NULL DEFAULT '[]',
                    document_ids_json TEXT NOT NULL DEFAULT '[]',
                    response_json TEXT
                );
                CREATE INDEX IF NOT EXISTS messages_by_conversation
                    ON messages(conversation_id, message_id);
            """)
            with connection:
                yield connection
        except (OSError, sqlite3.Error) as exc:
            raise ConversationError(f"Could not open conversation storage: {exc}") from exc
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _check_id(conversation_id: str):
        try:
            uuid.UUID(conversation_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ConversationNotFound("Conversation not found.") from exc

    @staticmethod
    def _message(row):
        return {
            "role": row["role"], "content": row["content"],
            "timestamp": row["timestamp"], "resolved_query": row["resolved_query"],
            "scope": json.loads(row["scope_json"]),
            "document_ids": json.loads(row["document_ids_json"]),
            "response": json.loads(row["response_json"]) if row["response_json"] else None,
        }

    def create(self, scope: list[str] | None = None) -> dict:
        scope = scope or []
        conversation_id = str(uuid.uuid4())
        now = _now()
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
                    (conversation_id, "New chat", now, now, json.dumps(scope)),
                )
        except sqlite3.Error as exc:
            raise ConversationError(f"Could not create conversation: {exc}") from exc
        return self.get(conversation_id)

    def list(self) -> list[dict]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT * FROM conversations ORDER BY updated_at DESC, conversation_id DESC"
                ).fetchall()
            return [{**dict(row), "scope": json.loads(row["scope_json"])} for row in rows]
        except sqlite3.Error as exc:
            raise ConversationError(f"Could not list conversations: {exc}") from exc

    def get(self, conversation_id: str) -> dict:
        self._check_id(conversation_id)
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,)
                ).fetchone()
                if row is None:
                    raise ConversationNotFound("Conversation not found.")
                messages = connection.execute(
                    "SELECT * FROM messages WHERE conversation_id = ? ORDER BY message_id",
                    (conversation_id,),
                ).fetchall()
            return {
                "conversation_id": row["conversation_id"], "title": row["title"],
                "created_at": row["created_at"], "updated_at": row["updated_at"],
                "scope": json.loads(row["scope_json"]),
                "messages": [self._message(message) for message in messages],
            }
        except sqlite3.Error as exc:
            raise ConversationError(f"Could not load conversation: {exc}") from exc

    def update_scope(self, conversation_id: str, scope: list[str]) -> dict:
        self._check_id(conversation_id)
        try:
            with self._connect() as connection:
                updated = connection.execute(
                    "UPDATE conversations SET scope_json = ?, updated_at = ? WHERE conversation_id = ?",
                    (json.dumps(scope), _now(), conversation_id),
                )
                if updated.rowcount == 0:
                    raise ConversationNotFound("Conversation not found.")
        except sqlite3.Error as exc:
            raise ConversationError(f"Could not update conversation scope: {exc}") from exc
        return self.get(conversation_id)

    def delete(self, conversation_id: str) -> None:
        self._check_id(conversation_id)
        try:
            with self._connect() as connection:
                deleted = connection.execute(
                    "DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,)
                )
                if deleted.rowcount == 0:
                    raise ConversationNotFound("Conversation not found.")
        except sqlite3.Error as exc:
            raise ConversationError(f"Could not delete conversation: {exc}") from exc

    def append_exchange(
        self, conversation_id: str, question: str, answer: str, *,
        scope: list[str], resolved_query: str, document_ids: list[str], response: dict,
    ) -> dict:
        """Commit a user/assistant turn and scope together, or not at all."""
        self._check_id(conversation_id)
        now = _now()
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT title FROM conversations WHERE conversation_id = ?", (conversation_id,)
                ).fetchone()
                if row is None:
                    raise ConversationNotFound("Conversation not found.")
                first = connection.execute(
                    "SELECT 1 FROM messages WHERE conversation_id = ? LIMIT 1", (conversation_id,)
                ).fetchone() is None
                connection.execute(
                    """INSERT INTO messages
                       (conversation_id, role, content, timestamp, resolved_query, scope_json, document_ids_json)
                       VALUES (?, 'user', ?, ?, ?, ?, ?)""",
                    (conversation_id, question, now, resolved_query,
                     json.dumps(scope), json.dumps(document_ids)),
                )
                connection.execute(
                    """INSERT INTO messages
                       (conversation_id, role, content, timestamp, scope_json, document_ids_json, response_json)
                       VALUES (?, 'assistant', ?, ?, ?, ?, ?)""",
                    (conversation_id, answer, now, json.dumps(scope), json.dumps(document_ids),
                     json.dumps(response)),
                )
                connection.execute(
                    "UPDATE conversations SET title = ?, scope_json = ?, updated_at = ? WHERE conversation_id = ?",
                    (short_title(question) if first else row["title"], json.dumps(scope),
                     now, conversation_id),
                )
        except (sqlite3.Error, TypeError) as exc:
            raise ConversationError(f"Could not save conversation: {exc}") from exc
        return self.get(conversation_id)
