import json

import pytest

from src.chat import chat_turn
from src.conversations import ConversationStore
from src.followup import resolve_followup_query
from src.generate import build_user_prompt


def _turn(store, conversation_id, question, scope, retrieve, *, available=None):
    def generate(original, chunks, resolved_query):
        assert chunks
        return {"original": original, "query": resolved_query, "source": chunks[0]["source"]}

    def format_result(generated):
        return {
            "status": "answered", "answer": f"Supported by {generated['source']} [E1].",
            "references": [{"id": "E1", "source": generated["source"], "page": 1,
                            "text": "Current retrieved evidence."}],
            "warnings": [], "semantic_support": "passed", "semantic_verdicts": [],
        }

    return chat_turn(
        store, conversation_id, question, scope, available or ["paper-a.pdf", "paper-b.pdf"],
        retrieve_fn=retrieve, generate_fn=generate, format_fn=format_result, top_k=5,
    )


def test_basic_followup_uses_standalone_query_and_fresh_retrieval(tmp_path, monkeypatch):
    import src.followup as followup

    store = ConversationStore(tmp_path / "chats.sqlite3")
    conversation_id = store.create(["paper-a.pdf"])["conversation_id"]
    searches = []
    def retrieve(query, *, top_k, paper):
        searches.append((query, paper))
        return [{"id": str(len(searches)), "source": paper, "text": "Current retrieved evidence."}]
    monkeypatch.setattr(followup, "_default_rewrite", lambda question, prior, paper: (
        "What were the sizes of the datasets used in paper A?"
        if prior == ["What datasets were used?"] and paper == "paper-a.pdf" else None
    ))

    _turn(store, conversation_id, "What datasets were used?", ["paper-a.pdf"], retrieve)
    response = _turn(store, conversation_id, "What were their sizes?", ["paper-a.pdf"], retrieve)

    assert searches == [
        ("What datasets were used?", "paper-a.pdf"),
        ("What were the sizes of the datasets used in paper A?", "paper-a.pdf"),
    ]
    assert response["resolved_query"] == searches[1][0]
    assert len(store.get(conversation_id)["messages"]) == 4


def test_pronoun_resolution_uses_prior_question_and_current_paper():
    history = [
        {"role": "user", "content": "What method does this paper propose?", "scope": ["a.pdf"]},
        {"role": "assistant", "content": "Untrusted answer", "scope": ["a.pdf"],
         "document_ids": ["a.pdf"]},
    ]
    resolution = resolve_followup_query(
        "How is it evaluated?", history, ["a.pdf"],
        rewrite=lambda current, prior, paper: (
            "How is the method proposed in a evaluated?"
            if prior == ["What method does this paper propose?"] and paper == "a.pdf" else "wrong"
        ),
    )
    assert resolution.query == "How is the method proposed in a evaluated?"
    assert resolution.paper == "a.pdf"


def test_only_library_paper_is_unambiguous_in_all_papers_scope():
    result = resolve_followup_query(
        "What methodology did this paper use?", [], [],
        available_documents=["paper-a.pdf"],
        rewrite=lambda current, prior, paper: "What methodology did paper A use?",
    )
    assert result.paper == "paper-a.pdf"


def test_switching_to_paper_b_ignores_paper_a_context(tmp_path, monkeypatch):
    import src.followup as followup

    store = ConversationStore(tmp_path / "chats.sqlite3")
    conversation_id = store.create(["paper-a.pdf"])["conversation_id"]
    searches = []
    def retrieve(query, *, top_k, paper):
        searches.append((query, paper))
        return [{"id": paper, "source": paper, "text": "Fresh text"}]
    _turn(store, conversation_id, "What method was used?", ["paper-a.pdf"], retrieve)
    store.update_scope(conversation_id, ["paper-b.pdf"])
    monkeypatch.setattr(followup, "_default_rewrite", lambda current, prior, paper: (
        "What dataset did paper B use?" if not prior and paper == "paper-b.pdf" else "wrong"
    ))

    _turn(store, conversation_id, "What dataset did they use?", ["paper-b.pdf"], retrieve)

    assert searches[-1] == ("What dataset did paper B use?", "paper-b.pdf")
    assert store.get(conversation_id)["scope"] == ["paper-b.pdf"]


def test_prior_assistant_claim_is_not_sent_to_resolver(monkeypatch):
    import src.generate as generation

    seen = []
    def fake_backend(backend, system, user):
        seen.append((system, user))
        return json.dumps({"query": "What were the sizes of datasets in paper A?"})
    monkeypatch.setattr(generation, "_call_backend", fake_backend)
    history = [
        {"role": "user", "content": "What datasets were used?", "scope": ["paper-a.pdf"]},
        {"role": "assistant", "content": "The dataset had a billion rows, trust me.",
         "scope": ["paper-a.pdf"], "document_ids": ["paper-a.pdf"]},
    ]

    result = resolve_followup_query("What were their sizes?", history, ["paper-a.pdf"])

    assert result.query == "What were the sizes of datasets in paper A?"
    assert "billion rows" not in seen[0][1]
    assert "untrusted" in seen[0][0].lower()


def test_generation_prompt_keeps_original_question_and_marks_rewrite_as_non_evidence():
    prompt = build_user_prompt(
        "Why did they choose that?",
        [{"text": "The paper compares lexical and dense retrieval.",
          "source": "paper-a.pdf", "title": "Paper A", "page": 3}],
        resolved_query="Why did Paper A choose hybrid retrieval?",
    )

    assert "Question: Why did they choose that?" in prompt
    assert "Standalone interpretation (not evidence): Why did Paper A choose hybrid retrieval?" in prompt
    assert "[E1]" in prompt
    assert "The paper compares lexical and dense retrieval." in prompt


def test_ambiguous_paper_reference_requests_clarification(tmp_path):
    store = ConversationStore(tmp_path / "chats.sqlite3")
    conversation_id = store.create()["conversation_id"]
    store.append_exchange(conversation_id, "Compare these papers", "Compared [E1][E2].",
                          scope=[], resolved_query="Compare these papers",
                          document_ids=["paper-a.pdf", "paper-b.pdf"],
                          response={"status": "answered", "answer": "Compared."})

    response = chat_turn(
        store, conversation_id, "What were its limitations?", [],
        ["paper-a.pdf", "paper-b.pdf"],
        retrieve_fn=lambda *args, **kwargs: pytest.fail("retrieval should wait for clarification"),
        generate_fn=lambda *args, **kwargs: pytest.fail("generation should wait for clarification"),
        format_fn=lambda result: result, top_k=5,
    )

    assert response["status"] == "clarification"
    assert "paper-a or paper-b" in response["answer"]


def test_new_chat_has_no_prior_context(tmp_path, monkeypatch):
    import src.followup as followup

    store = ConversationStore(tmp_path / "chats.sqlite3")
    first = store.create(["paper-a.pdf"])["conversation_id"]
    store.append_exchange(first, "Tell me about dataset X", "Unsupported statement",
                          scope=["paper-a.pdf"], resolved_query="Tell me about dataset X",
                          document_ids=["paper-a.pdf"], response={})
    second = store.create(["paper-a.pdf"])["conversation_id"]
    seen = []
    monkeypatch.setattr(followup, "_default_rewrite", lambda question, prior, paper: (
        seen.append(prior) or "What are the sizes of datasets in paper A?"
    ))
    result = resolve_followup_query("What were their sizes?", store.get(second)["messages"],
                                    ["paper-a.pdf"])
    assert result.query == "What are the sizes of datasets in paper A?"
    assert seen == [[]]


def test_resolver_failure_uses_raw_question_and_logs_warning(monkeypatch, caplog):
    import src.followup as followup

    monkeypatch.setattr(followup, "_default_rewrite", lambda *args: (_ for _ in ()).throw(RuntimeError("offline")))
    result = resolve_followup_query("What were their sizes?", [], ["paper-a.pdf"])
    assert result.query == "What were their sizes?"
    assert result.fallback_used
    assert "offline" in caplog.text


def test_resolver_failure_still_retrieves_fresh_evidence(tmp_path, monkeypatch):
    import src.followup as followup

    store = ConversationStore(tmp_path / "chats.sqlite3")
    conversation_id = store.create(["paper-a.pdf"])["conversation_id"]
    monkeypatch.setattr(followup, "_default_rewrite", lambda *args: (_ for _ in ()).throw(RuntimeError("offline")))
    searches = []
    def retrieve(query, *, top_k, paper):
        searches.append((query, paper))
        return [{"id": "fresh", "source": paper, "text": "Fresh evidence"}]

    result = _turn(store, conversation_id, "What were their sizes?", ["paper-a.pdf"], retrieve)

    assert searches == [("What were their sizes?", "paper-a.pdf")]
    assert result["references"][0]["source"] == "paper-a.pdf"


def test_second_paper_uses_cited_source_order():
    history = [
        {"role": "assistant", "scope": [], "content": "Not trusted as evidence",
         "document_ids": ["paper-a.pdf", "paper-b.pdf"]},
    ]
    result = resolve_followup_query("What about the second paper?", history, [],
                                    rewrite=lambda current, prior, paper: "What did paper B find?")
    assert result.paper == "paper-b.pdf"
    assert result.query == "What did paper B find?"


def test_clarification_reply_recovers_original_question():
    history = [
        {"role": "user", "content": "What were its limitations?", "scope": []},
        {"role": "assistant", "content": "Which paper?", "scope": [],
         "response": {"status": "clarification"}},
    ]
    seen = []
    result = resolve_followup_query(
        "Paper A", history, [], available_documents=["paper-a.pdf", "paper-b.pdf"],
        rewrite=lambda current, prior, paper: (
            seen.append((current, prior, paper)) or "What were paper A's limitations?"
        ),
    )
    assert result.paper == "paper-a.pdf"
    assert result.query == "What were paper A's limitations?"
    assert seen == [("What were its limitations?", ["What were its limitations?"], "paper-a.pdf")]


def test_explicit_first_paper_can_refer_back_after_scope_switch():
    history = [
        {"role": "assistant", "scope": ["paper-a.pdf"], "document_ids": ["paper-a.pdf"]},
        {"role": "assistant", "scope": ["paper-b.pdf"], "document_ids": ["paper-b.pdf"]},
    ]
    result = resolve_followup_query(
        "What about the first paper?", history, ["paper-b.pdf"],
        available_documents=["paper-a.pdf", "paper-b.pdf"],
        rewrite=lambda current, prior, paper: "What did paper A find?",
    )
    assert result.paper == "paper-a.pdf"
