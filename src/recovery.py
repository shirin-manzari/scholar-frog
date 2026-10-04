"""One query-local evidence recovery, independent of generation retries."""
import json
import os
from collections import Counter


def recovery_query(question, verdicts):
    """Retain the original verbatim, including names, units, numbers and negation.

    Unsupported claims are explicitly questions to investigate, never asserted facts.
    The verifier's prose is not used as a retrieval instruction.
    """
    gaps = [v['claim'] for v in verdicts if v.get('supported') is False and v.get('claim', '').strip()]
    if not gaps:
        return None
    return question + '\nEvidence gap to check, not an established fact: ' + json.dumps(gaps[0], ensure_ascii=False)


def combine_evidence(question, original, additional, anchor_limit):
    """Append without changing E IDs; reject duplicate/overlapping versioned spans."""
    from src.generate import SYSTEM_PROMPT, build_context, build_user_prompt
    from src.context_budget import generation_token_counter, positive_setting
    from src.retrieve import _non_negative_int
    result = list(original)
    counter = generation_token_counter()
    def anchors(rows):
        return {h['id'] for c in rows for h in c.get('anchor_hits', [])} | {
            c.get('id', c.get('text')) for c in rows if 'anchor_hits' not in c}
    def same(a, b):
        ma, mb = a.get('metadata', {}), b.get('metadata', {})
        if a.get('id') is not None and a.get('id') == b.get('id'):
            return True
        version_a = (ma.get('document_id', a.get('source')), ma.get('document_version'), a.get('page'))
        version_b = (mb.get('document_id', b.get('source')), mb.get('document_version'), b.get('page'))
        if version_a != version_b:
            return False
        if a['text'] == b['text']:
            return True
        if all(k in m for m in (ma, mb) for k in ('character_start', 'character_end')):
            return max(ma['character_start'], mb['character_start']) < min(ma['character_end'], mb['character_end'])
        return False
    original_expansion = max((c.get('metadata', {}).get('context_usage', {}).get('expansion_tokens', 0) for c in original), default=0)
    added_expansion = 0
    for c in additional:
        if any(same(c, old) for old in result):
            continue
        trial = result + [c]
        # Conservatively charge new expanded blocks entirely to expansion allowance.
        expansion = counter.count(build_context([c])) if c.get('evidence_kind') in {'anchor_context', 'expansion'} else 0
        if original_expansion + added_expansion + expansion > _non_negative_int('CONTEXT_EXPANSION_TOKENS', 1500):
            continue
        scope = original[0].get('metadata', {}).get('retrieval_scope', {}) if original else {}
        policy = os.getenv('PAPER_CAP_POLICY', 'baseline')
        plan = scope.get('plan', {})
        if policy == 'baseline' or (plan.get('task') == 'focused' and len(plan.get('papers', ())) != 1):
            from src.retrieve import _positive_int
            per_paper = Counter()
            for item in trial:
                paper = item.get('metadata', {}).get('document_id', item.get('source'))
                per_paper[paper] += len(item.get('anchor_hits', [item]))
            if any(n > _positive_int('MAX_CHUNKS_PER_PAPER', 2) for n in per_paper.values()):
                continue
        if len(anchors(trial)) > anchor_limit:
            continue
        if counter.count(build_context(trial)) > positive_setting('CONTEXT_BUDGET_TOKENS', 4000):
            continue
        required = counter.prompt_tokens(SYSTEM_PROMPT, build_user_prompt(question, trial), positive_setting('CONTEXT_FRAMING_RESERVE', 32))
        required += positive_setting('CONTEXT_ANSWER_RESERVE', 1024) + _non_negative_int('CONTEXT_RETRY_RESERVE', 256)
        if required > positive_setting('CONTEXT_WINDOW_TOKENS', 8192):
            continue
        result.append(c); added_expansion += expansion
    return result
