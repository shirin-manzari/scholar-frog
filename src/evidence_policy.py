"""Task allocations over relevant anchors only; context never spends paper quotas."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class RetrievalPlan:
    task: str = 'focused'
    papers: tuple[str, ...] = ()  # Explicit source paths, never model-inferred scope.
    dimensions: tuple[str, ...] = ()
    subqueries: tuple[str, ...] = ()
    positions: tuple[str, ...] = ()  # Include study-condition terms when analyzing conflicts.


def infer_plan(question, papers=()):
    if re.search(r'\b(conflict|contradict|disagree)', question, re.I):
        task = 'conflict'
    elif re.search(r'\b(compare|comparison|differ)', question, re.I):
        task = 'comparison'
    elif re.search(r'\b(summarize|summarise|summary|overview)\b', question, re.I):
        task = 'summary'
    else:
        task = 'focused'
    # Use only literal requested dimensions after an explicit delimiter.
    match = re.search(r'\b(?:on|across|regarding)\s+(.+?)[?.]?$' , question, re.I)
    dimensions = tuple(x.strip() for x in re.split(r',|\band\b', match[1]) if x.strip()) if match and task in {'comparison', 'conflict'} else ()
    return RetrievalPlan(task, tuple(papers), dimensions)


def select_anchors(candidates, plan, budget, cap, paper_key, debug=None, fits=None):
    if plan.task not in {'focused', 'summary', 'comparison', 'conflict'}:
        raise ValueError('Unknown retrieval task')
    selected, seen, counts = [], set(), {}
    rejected = []
    papers = list(plan.papers) or list(dict.fromkeys(paper_key(c) for c in candidates))
    key = lambda c: c['source'] if plan.papers else paper_key(c)
    candidates = [c for c in candidates if key(c) in papers]
    # Single-paper focus and summaries may spend the full anchor allowance.
    limit = budget if len(papers) == 1 or plan.task != 'focused' else cap
    labels = plan.positions if plan.task == 'conflict' and plan.positions else plan.dimensions
    def covers(c, label):
        tokens = re.findall(r'\w+', label.casefold())
        text = (c.get('section', '') + ' ' + c['text']).casefold()
        def matches(value):
            return bool(tokens) and all(re.search(r'(?<!\w)' + re.escape(t) + r'(?!\w)', value) for t in tokens)
        if plan.task == 'conflict' and plan.positions:
            # Literal condition coverage must not turn a negated result into a
            # positive position. Semantic interpretation remains the verifier's job.
            negations = {'no', 'not', 'never', 'without', 'neither'}
            return any(matches(sentence) and (negations.intersection(tokens)
                or not negations.intersection(re.findall(r'\w+', sentence)))
                for sentence in re.split(r'[.!?;]\s+|\n+', text))
        return matches(text)
    def add(c):
        p = key(c)
        if c['id'] not in seen and len(selected) < budget and counts.get(p, 0) < limit:
            if fits is not None and not fits(selected + [c]):
                if c['id'] not in {row['id'] for row in rejected}:
                    rejected.append({'id': c['id'], 'reason': 'context_budget'})
                return False
            selected.append(c); seen.add(c['id']); counts[p] = counts.get(p, 0) + 1
            return True
        return False
    missing = []
    pools = []
    for p in papers:
        hits = [c for c in candidates if key(c) == p]
        buckets = labels or (tuple(dict.fromkeys(c.get('section', 'unknown') for c in hits))
                             if plan.task == 'summary' else ('all',))
        if not hits:
            missing.append({'paper': p, 'dimension': None, 'reason': 'no_relevant_evidence'})
        for label in buckets:
            pool = hits if label == 'all' else [c for c in hits if (c.get('section', 'unknown') == label if plan.task == 'summary' else covers(c, label))]
            if not pool:
                missing.append({'paper': p, 'dimension': label, 'reason': 'no_relevant_evidence'})
            pools.append((p, label, pool))
    # Dimension-major, paper-minor rotation balances both within scarce budgets.
    pools.sort(key=lambda row: (list(labels).index(row[1]) if row[1] in labels else 0, papers.index(row[0])))
    if debug is not None:
        debug.update(task=plan.task, missing_allocations=missing, anchor_budget=budget)
    while any(pool for _, _, pool in pools) and len(selected) < budget:
        progressed = False
        for _, _, pool in pools:
            while pool:
                if add(pool.pop(0)):
                    progressed = True; break
        if not progressed:
            break
    # Redistribution occurs only after gaps have been recorded, never below threshold.
    for c in candidates:
        add(c)
    if debug is not None:
        debug['budget_rejected_candidates'] = rejected
        debug['paper_allocations'] = {p: counts.get(p, 0) for p in papers}
        debug['unselected_allocations'] = [{'paper': p, 'dimension': d, 'reason': 'anchor_budget'} for p, d, pool in pools if pool and not any(key(c) == p and (d == 'all' or covers(c, d)) for c in selected)]
        debug['dimension_coverage'] = {p: {d: sum(covers(c, d) for c in selected if key(c) == p) for d in labels} for p in papers}
    return selected
