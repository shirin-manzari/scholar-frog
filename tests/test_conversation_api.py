"""Exercise conversation HTTP routes without opening a listening socket."""

import io
import json

import app
from src.conversations import ConversationStore


def _request(method, path, payload=None):
    handler = object.__new__(app.Handler)
    body = json.dumps(payload).encode() if payload is not None else b""
    handler.command = method
    handler.path = path
    handler.headers = {
        "Host": "127.0.0.1:8765",
        "Content-Length": str(len(body)),
    }
    handler.rfile = io.BytesIO(body)
    output = []
    handler.send_json = lambda data, status=200: output.append((status, data))
    getattr(handler, f"do_{method}")()
    assert len(output) == 1
    return output[0]


def test_conversation_routes_create_list_load_scope_ask_delete(tmp_path, monkeypatch):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "a.pdf").write_bytes(b"%PDF-")
    monkeypatch.setattr(app, "PAPERS", papers)
    monkeypatch.setattr(app, "CONVERSATIONS", ConversationStore(tmp_path / "chats.sqlite3"))

    status, created = _request("POST", "/api/conversations", {"document_ids": ["a.pdf"]})
    assert status == 201
    conversation_id = created["conversation_id"]
    assert created["scope"] == ["a.pdf"]

    status, listing = _request("GET", "/api/conversations")
    assert status == 200
    assert listing["conversations"][0]["conversation_id"] == conversation_id

    status, updated = _request("PATCH", f"/api/conversations/{conversation_id}",
                               {"document_ids": []})
    assert status == 200
    assert updated["scope"] == []

    observed = []
    def fake_ask(chat_id, question, document_ids):
        observed.append((chat_id, question, document_ids))
        return {"conversation_id": chat_id, "answer": "Supported [E1].",
                "references": [{"id": "E1"}]}
    monkeypatch.setattr(app, "ask_conversation_question", fake_ask)
    status, answer = _request("POST", "/api/ask", {
        "conversation_id": conversation_id, "question": "What method?",
        "document_ids": ["a.pdf"],
    })
    assert status == 200
    assert answer["references"] == [{"id": "E1"}]
    assert observed == [(conversation_id, "What method?", ["a.pdf"])]

    status, loaded = _request("GET", f"/api/conversations/{conversation_id}")
    assert status == 200
    assert loaded["scope"] == []

    status, deleted = _request("DELETE", f"/api/conversations/{conversation_id}")
    assert status == 200
    assert deleted["deleted"] == conversation_id
    assert _request("GET", f"/api/conversations/{conversation_id}")[0] == 404


def test_conversation_api_rejects_removed_or_duplicate_papers(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "PAPERS", tmp_path)
    monkeypatch.setattr(app, "CONVERSATIONS", ConversationStore(tmp_path / "chats.sqlite3"))
    assert _request("POST", "/api/conversations", {"document_ids": ["gone.pdf"]})[0] == 400
    (tmp_path / "a.pdf").write_bytes(b"%PDF-")
    assert _request("POST", "/api/conversations", {"document_ids": ["a.pdf", "a.pdf"]})[0] == 400
