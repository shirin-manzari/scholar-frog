import json

import pytest

from src.conversations import ConversationError, ConversationNotFound, ConversationStore


def test_conversation_persists_across_store_instances(tmp_path):
    path = tmp_path / "chats.sqlite3"
    store = ConversationStore(path)
    created = store.create(["paper-a.pdf"])
    conversation_id = created["conversation_id"]
    store.append_exchange(
        conversation_id, "What datasets were used?", "A cited answer [E1].",
        scope=["paper-a.pdf"], resolved_query="What datasets were used in paper A?",
        document_ids=["paper-a.pdf"],
        response={"status": "answered", "answer": "A cited answer [E1].",
                  "references": [{"id": "E1", "source": "paper-a.pdf"}]},
    )

    restored = ConversationStore(path).get(conversation_id)

    assert restored["title"] == "What datasets were used"
    assert restored["scope"] == ["paper-a.pdf"]
    assert [message["role"] for message in restored["messages"]] == ["user", "assistant"]
    assert restored["messages"][0]["resolved_query"] == "What datasets were used in paper A?"
    assert restored["messages"][1]["response"]["references"][0]["id"] == "E1"


def test_new_chat_is_independent_and_deletion_is_scoped(tmp_path):
    store = ConversationStore(tmp_path / "chats.sqlite3")
    first = store.create()["conversation_id"]
    store.append_exchange(first, "Question A", "Answer A", scope=[],
                          resolved_query="Question A", document_ids=[], response={})
    second = store.create()["conversation_id"]

    assert store.get(second)["messages"] == []
    store.delete(first)
    with pytest.raises(ConversationNotFound):
        store.get(first)
    assert store.get(second)["conversation_id"] == second


def test_failed_exchange_does_not_leave_half_a_turn(tmp_path):
    store = ConversationStore(tmp_path / "chats.sqlite3")
    conversation_id = store.create()["conversation_id"]

    with pytest.raises(ConversationError, match="Could not save conversation"):
        store.append_exchange(
            conversation_id, "Question", "Answer", scope=[], resolved_query="Question",
            document_ids=[], response={"bad": object()},
        )

    assert store.get(conversation_id)["messages"] == []
