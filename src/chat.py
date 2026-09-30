"""Conversation-aware orchestration around the existing RAG pipeline."""

from __future__ import annotations

import logging
from collections import OrderedDict

from src.followup import resolve_followup_query


logger = logging.getLogger(__name__)


def _search(query, scope, top_k, retrieve_fn):
    if len(scope) <= 1:
        return retrieve_fn(query, top_k=top_k, paper=scope[0] if scope else None)
    # Reuse the existing per-paper retrieval path for callers selecting several
    # papers. Round-robin keeps one paper from consuming all available slots.
    batches = [retrieve_fn(query, top_k=top_k, paper=paper) for paper in scope]
    results = OrderedDict()
    for position in range(top_k):
        for batch in batches:
            if position < len(batch):
                item = batch[position]
                key = item.get("id", (item.get("source"), item.get("page"), item.get("text")))
                results.setdefault(key, item)
                if len(results) >= top_k:
                    return list(results.values())
    return list(results.values())


def chat_turn(
    store, conversation_id, question, selected_documents, available_documents, *,
    retrieve_fn, generate_fn, format_fn, top_k,
):
    """Resolve, retrieve fresh evidence, generate, validate, then save one turn."""
    conversation = store.get(conversation_id)
    scope = list(conversation["scope"] if selected_documents is None else selected_documents)
    missing = [paper for paper in scope if paper not in available_documents]
    if missing:
        raise ValueError("Selected paper is no longer in the library. Choose another paper.")

    if not available_documents:
        response = {
            "status": "no_papers",
            "answer": "You gave frog no papers. add a PDF and sync the library first.",
            "references": [], "warnings": [],
            "semantic_support": "not_checked", "semantic_verdicts": [],
        }
        resolved_query, document_ids = question, []
    else:
        resolution = resolve_followup_query(
            question, conversation["messages"], scope,
            available_documents=available_documents,
        )
        resolved_query = resolution.query
        if resolution.fallback_used:
            logger.warning("Follow-up resolver failed; using raw query for conversation_id=%s", conversation_id)
        used_scope = [resolution.paper] if resolution.paper else scope
        logger.info(
            "chat query conversation_id=%s original_query=%r resolved_query=%r "
            "selected_document_ids=%s retrieval_scope=%s resolver_fallback_used=%s "
            "clarification_requested=%s",
            conversation_id, question, resolved_query, scope, used_scope,
            resolution.fallback_used, bool(resolution.clarification),
        )
        if resolution.clarification:
            response = {
                "status": "clarification", "answer": resolution.clarification,
                "references": [], "warnings": [], "semantic_support": "not_checked",
                "semantic_verdicts": [],
            }
            document_ids = []
        else:
            chunks = _search(resolved_query, used_scope, top_k, retrieve_fn)
            generated = generate_fn(question, chunks, resolved_query=resolved_query)
            response = format_fn(generated)
            # Only cited sources establish what was actually discussed. Search
            # candidates from unrelated papers must not create false referents.
            document_ids = list(dict.fromkeys(
                item.get("source") for item in response.get("references", [])
                if item.get("source") in available_documents
            ))
            if not document_ids and len(used_scope) == 1:
                document_ids = list(used_scope)

    response["conversation_id"] = conversation_id
    response["resolved_query"] = resolved_query
    store.append_exchange(
        conversation_id, question, response["answer"], scope=scope,
        resolved_query=resolved_query, document_ids=document_ids, response=response,
    )
    return response
