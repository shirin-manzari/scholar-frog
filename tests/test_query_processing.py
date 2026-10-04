"""Query transformations, bounded retrieval, and provenance without model downloads."""
from dataclasses import replace

import pytest

from src import ingest, retrieve as r, sync
from src.generate import build_user_prompt
from src.index_config import IndexConfig, check_compatibility, new_metadata
from src.sync import CommittedSnapshot
from test_retrieve import FakeCollection, FakeEmbeddingModel, committed_snapshot
from test_passages import corpus_for, expand


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ('QUERY_INSTRUCTION', 'QUERY_EXPANSION', 'QUERY_MAX_VARIANTS'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(r, '_bm25_cache', None)
    monkeypatch.setattr(r, '_terminology_cache', None)


def record(text, doc='a', chunk='a'):
    return {'id': chunk, 'text': text, 'distance': .1,
            'metadata': {'source': doc + '.pdf', 'title': 'Paper', 'page': 1,
                         'document_id': doc, 'document_version': 'v1'}}


def setup_pipeline(monkeypatch, rows):
    collection = FakeCollection(rows)
    monkeypatch.setattr(r, 'get_collection', lambda: collection)
    monkeypatch.setattr(r, '_get_committed_snapshot', committed_snapshot)
    model = FakeEmbeddingModel()
    monkeypatch.setattr(r, 'get_embedding_model', lambda: model)
    monkeypatch.setattr(r, 'get_index_config', lambda: IndexConfig(embedding_dimension=2))
    monkeypatch.setattr(r, 'expand_context', lambda q, anchors, corpus, snapshot: anchors)
    return collection, model


@pytest.mark.parametrize('model,mode,expected', [
    ('BAAI/bge-small-en-v1.5', 'auto', r.BGE_QUERY_INSTRUCTION),
    ('BAAI/bge-small-en-v1.5', 'off', ''),
    ('BAAI/bge-m3', 'auto', ''),
    ('sentence-transformers/all-MiniLM-L6-v2', 'auto', ''),
])
def test_instruction_model_policy(monkeypatch, model, mode, expected):
    monkeypatch.setenv('QUERY_INSTRUCTION', mode)
    assert r.query_instruction(model) == expected


def test_instruction_only_reaches_dense_encoder(monkeypatch):
    c, model = setup_pipeline(monkeypatch, [record('Retrieval augmented generation (RAG).')])
    monkeypatch.setenv('QUERY_EXPANSION', 'true')
    encoded, lexical, reranked = [], [], []
    original_encode = model.encode
    monkeypatch.setattr(model, 'encode', lambda queries, **kw: encoded.extend(queries) or original_encode(queries, **kw))
    bm25 = r._bm25_search
    monkeypatch.setattr(r, '_bm25_search', lambda query, *a: lexical.append(query) or bm25(query, *a))
    monkeypatch.setattr(r, '_rerank', lambda q, candidates, name: reranked.append(q) or [dict(c, reranker_score=.9) for c in candidates])
    debug = {}
    hits = r.retrieve('Is RAG reliable?', debug=debug)
    assert encoded == [r.BGE_QUERY_INSTRUCTION + q for q in lexical]
    assert lexical == ['Is RAG reliable?', 'Is RAG (Retrieval augmented generation) reliable?']
    assert reranked == ['Is RAG reliable?']
    prompt = build_user_prompt('Is RAG reliable?', hits)
    assert 'Question: Is RAG reliable?' in prompt
    assert r.BGE_QUERY_INSTRUCTION not in prompt
    assert debug['original_query'] == 'Is RAG reliable?'
    assert debug['instruction_used']
    assert len(debug['candidate_provenance']['a']['queries']) == 4
    assert debug['reranked_scores']['a'] == .9
    assert {'text', 'source', 'title', 'page', 'distance'} <= hits[0].keys()
    assert {p['query'] for p in hits[0]['retrieval_queries']} == set(lexical)


def test_query_capacity_counts_instruction_and_special_tokens(monkeypatch):
    c, model = setup_pipeline(monkeypatch, [record('Text')])
    model.max_seq_length = len(r.BGE_QUERY_INSTRUCTION) + 5 + 2
    s = committed_snapshot(c)
    assert r._dense_search('12345', c, 1, 1, s)
    with pytest.raises(ValueError, match='including its query instruction'):
        r._dense_search('123456', c, 1, 1, s)
    monkeypatch.setenv('QUERY_INSTRUCTION', 'off')
    assert r._dense_search('123456', c, 1, 1, s)


def test_document_embeddings_and_fingerprint_unchanged(monkeypatch, tmp_path):
    config = IndexConfig(embedding_dimension=2, chunk_size=60, chunk_overlap=10)
    model = FakeEmbeddingModel()
    calls = []
    original = model.encode
    monkeypatch.setattr(model, 'encode', lambda documents, **kw: calls.append(list(documents)) or original(documents, **kw))
    monkeypatch.setattr(ingest, 'get_embedding_model', lambda: model)
    monkeypatch.setattr(ingest, 'get_index_config', lambda: config)
    monkeypatch.setattr(ingest, 'extract_pages', lambda path: [(1, 'Document content is unchanged.')])
    paper = tmp_path / 'paper.pdf'
    paper.write_text('stub')
    monkeypatch.setenv('QUERY_INSTRUCTION', 'off')
    before = sync._prepare({'absolute': paper, 'relative': 'paper.pdf'}, 'hash', ['paper.pdf'])
    monkeypatch.setenv('QUERY_INSTRUCTION', 'auto')
    monkeypatch.setenv('QUERY_EXPANSION', 'true')
    after = sync._prepare({'absolute': paper, 'relative': 'paper.pdf'}, 'hash', ['paper.pdf'])
    assert before == after
    assert calls == [before['documents'], before['documents']]
    assert check_compatibility(config, new_metadata(config)) == 'compatible'
    assert all(r.BGE_QUERY_INSTRUCTION not in doc for call in calls for doc in call)


def test_acronyms_expanded_terms_and_function_words(monkeypatch):
    c = FakeCollection([record('Retrieval-Augmented Generation (RAG). RLHF (reinforcement learning from human feedback).')])
    s = committed_snapshot(c)
    _, corpus, _ = r._get_bm25_index(c, s)
    terms = r._get_terminology(c, s, corpus)
    assert terms == {'RAG': 'Retrieval Augmented Generation', 'RLHF': 'reinforcement learning from human feedback'}
    monkeypatch.setenv('QUERY_EXPANSION', 'true')
    assert r._search_queries('Why use RAG?', 'Why use RAG?', terms)[1] == 'Why use RAG (Retrieval Augmented Generation)?'
    assert r._search_queries('Why use retrieval-augmented generation?', 'Why use retrieval-augmented generation?', terms)[1].endswith('generation (RAG)?')


def test_ambiguity_scope_pending_and_revision_invalidation(monkeypatch):
    c = FakeCollection([record('ABC (Alpha Beta Charlie).'),
                        record('ABC (Another Basic Concept).', 'b', 'b'),
                        record('XYZ (Xylophone Yellow Zebra).', 'pending', 'pending')])
    s0 = committed_snapshot(c)
    owners = {k: v for k, v in s0.owners.items() if k != 'pending'}
    s = CommittedSnapshot('revision1', frozenset(owners), owners)
    _, corpus, _ = r._get_bm25_index(c, s)
    assert r._get_terminology(c, s, corpus) == {}
    assert r._get_terminology(c, s, corpus, 'a') == {'ABC': 'Alpha Beta Charlie'}
    first_cache = r._terminology_cache
    assert r._get_terminology(c, s, corpus, 'b') == {'ABC': 'Another Basic Concept'}
    assert r._terminology_cache is first_cache
    # An uncommitted physical edit does not invalidate the committed cache.
    c.rows['pending']['text'] = 'XYZ (Xenon Yellow Zebra).'
    assert r._get_terminology(c, s, corpus) == {}
    after = CommittedSnapshot('revision2', s0.chunk_ids, s0.owners)
    _, corpus, _ = r._get_bm25_index(c, after)
    assert r._get_terminology(c, after, corpus)['XYZ'] == 'Xenon Yellow Zebra'
    assert r._terminology_cache is not first_cache


@pytest.mark.parametrize('question', [
    'Does RAG not reduce error at 20 mg/L in 2025?',
    'Compare RAG with GPT-4 for Nguyen and BGE-reranker-base at 0.01%.',
    'Is RAG no better than C++17 for Müller on 2026-10-04?',
])
def test_alias_insertion_preserves_all_original_characters(monkeypatch, question):
    monkeypatch.setenv('QUERY_EXPANSION', 'true')
    variants = r._search_queries(question, question, {'RAG': 'Retrieval Augmented Generation'})
    assert variants[0] == question
    assert len(variants) == 2
    assert variants[1].replace(' (Retrieval Augmented Generation)', '') == question
    assert r._search_queries(question.replace('RAG', 'RAG-2'), question.replace('RAG', 'RAG-2'), {'RAG': 'Retrieval Augmented Generation'}) == [question.replace('RAG', 'RAG-2')]


def test_corrections_keep_names_identifiers_and_known_rare_terms():
    corpus = [['dimension'] * 8 + ['trustworthiness'] * 8 + ['diminention']]
    q = 'Diminention and DiminentIon diminention-2 GPT-4 trustworthinesX diminention'
    assert r._correct_search_question(q, corpus) == q
    assert r._correct_search_question('diminention', [['dimension'] * 8]) == 'dimension'


def test_original_retained_and_shared_plan_correction_alias_limit(monkeypatch):
    monkeypatch.setenv('QUERY_EXPANSION', 'true')
    monkeypatch.setenv('QUERY_MAX_VARIANTS', '99')
    queries = r._search_queries('RAG?', 'RAG corrected?', {'RAG': 'Retrieval Augmented Generation'},
                               subqueries=['RAG?', 'RAG task?', 'RAG task?', 'Other task?'])
    assert queries == ['RAG?', 'RAG task?', 'Other task?']
    monkeypatch.setenv('QUERY_MAX_VARIANTS', '0')
    assert r._search_queries('RAG?', 'corrected', {'RAG': 'Retrieval Augmented Generation'}) == ['RAG?']
    monkeypatch.setenv('QUERY_MAX_VARIANTS', '2')
    monkeypatch.setenv('QUERY_EXPANSION', 'false')
    assert r._search_queries('RAG?', 'RAG?', {'RAG': 'Retrieval Augmented Generation'}) == ['RAG?']


@pytest.mark.parametrize('total', [1, 2, 3, 20, 51])
def test_shared_candidate_budgets(total):
    for n in (1, 2, 3):
        budgets = r._query_budgets(total, n)
        assert sum(budgets) == total
        assert budgets[0] >= sum(budgets[1:])


def test_weighted_fusion_caps_variants_and_keeps_per_query_scores():
    p = lambda q, method, score: [{'query': q, 'method': method, 'score': score, 'original': q == 'original'}]
    original = {'id': 'a', 'dense_distance': .3, 'dense_score': -.3, 'retrieval_queries': p('original', 'dense', -.3)}
    variants = [{'id': 'b', 'bm25_score': 20, 'retrieval_queries': p('v1', 'bm25', 20)}]
    fused = r.reciprocal_rank_fusion([original, original], variants, variants, weights=[1, .5, .5])
    assert len(fused) == 2
    assert fused[0]['rrf_score'] == fused[1]['rrf_score']
    merged = r.reciprocal_rank_fusion([original], [{**original, 'dense_distance': .1, 'dense_score': -.1, 'retrieval_queries': p('v1', 'dense', -.1)}])
    assert len(merged) == 1
    assert len(merged[0]['retrieval_queries']) == 2
    assert merged[0]['dense_distance'] == .1


def test_paper_scope_and_pending_exclusion_with_expansion(monkeypatch):
    c, model = setup_pipeline(monkeypatch, [record('ABC (Alpha Beta Charlie).'),
                                         record('ABC (Another Basic Concept).', 'b', 'b'),
                                         record('XYZ (Xenon Yellow Zebra).', 'a', 'pending')])
    s = committed_snapshot(c)
    s = CommittedSnapshot('committed', frozenset({'a', 'b'}), {k: v for k, v in s.owners.items() if k != 'pending'})
    monkeypatch.setattr(r, '_get_committed_snapshot', lambda c: s)
    monkeypatch.setenv('QUERY_EXPANSION', 'true')
    for mode in ('dense', 'hybrid', 'hybrid-rerank'):
        monkeypatch.setattr(r, '_rerank', lambda q, hits, name: [dict(h, reranker_score=.9) for h in hits])
        debug = {}
        hits = r.retrieve('ABC?', paper='a.pdf', retrieval_mode=mode, debug=debug)
        assert {h['id'] for h in hits} == {'a'}
        assert debug['variants'] == ['ABC (Alpha Beta Charlie)?']
        assert r._get_terminology(c, s, r._get_bm25_index(c, s)[1]).get('XYZ') is None


def test_context_provenance_survives_merging_and_citation_assignment():
    corpus = corpus_for([(1, 'First.\n\nSecond.')])
    for i, anchor in enumerate(corpus):
        anchor['retrieval_queries'] = [{'query': str(i), 'original': i == 0, 'method': 'dense'}]
    result = expand(corpus, corpus, radius=0)
    assert {p['query'] for p in result[0]['retrieval_queries']} == {'0', '1'}
    assert len(result[0]['anchor_hits']) == 2
    assert all(hit['retrieval_queries'] for hit in result[0]['anchor_hits'])


def test_corrected_query_is_additional_and_reranker_keeps_original(monkeypatch):
    c, model = setup_pipeline(monkeypatch, [record('dimension ' * 8)])
    encoded = []
    encode = model.encode
    monkeypatch.setattr(model, 'encode', lambda qs, **kw: encoded.extend(qs) or encode(qs, **kw))
    rerank_questions = []
    monkeypatch.setattr(r, '_rerank', lambda q, cs, name: rerank_questions.append(q) or [dict(c, reranker_score=.9) for c in cs])
    q = 'What is diminention?'
    assert r.retrieve(q)
    assert encoded == [r.BGE_QUERY_INSTRUCTION + q, r.BGE_QUERY_INSTRUCTION + 'What is dimension?']
    assert rerank_questions == [q]


def test_over_capacity_variant_does_not_discard_original(monkeypatch):
    c, model = setup_pipeline(monkeypatch, [record('Retrieval augmented generation (RAG).')])
    monkeypatch.setenv('QUERY_EXPANSION', 'true')
    model.max_seq_length = len(r.BGE_QUERY_INSTRUCTION) + len('RAG?') + 2
    debug = {}
    assert r.retrieve('RAG?', retrieval_mode='hybrid', debug=debug)
    assert debug['variants'] == []
    assert len(debug['rejected_variants']) == 1
    assert debug['candidate_counts'][0]['dense_budget'] == r.RetrievalConfig.from_env().dense_candidates


def test_original_candidates_reserved_when_variant_ranks_dominate(monkeypatch):
    c, model = setup_pipeline(monkeypatch, [record('dimension ' * 8)])
    monkeypatch.setenv('RERANK_CANDIDATES', '2')
    monkeypatch.setattr(r, '_search_queries', lambda *a: ['original', 'variant'])
    original = r._make_result('original-hit', 'Evidence', c.rows['a']['metadata'])
    variant = r._make_result('variant-hit', 'Alias evidence', c.rows['a']['metadata'])
    other = r._make_result('variant-other', 'Alias evidence', c.rows['a']['metadata'])
    monkeypatch.setattr(r, '_dense_search', lambda q, *a: [original] if q == 'original' else [variant, other])
    monkeypatch.setattr(r, '_bm25_search', lambda q, *a: [] if q == 'original' else [variant, other])
    seen = []
    monkeypatch.setattr(r, '_rerank', lambda q, cs, name: seen.extend(cs) or [dict(c, reranker_score=.9) for c in cs])
    r.retrieve('original')
    assert len(seen) == 2
    assert 'original-hit' in {c['id'] for c in seen}


@pytest.mark.parametrize('name,value', [('QUERY_INSTRUCTION', 'bge'), ('QUERY_EXPANSION', 'maybe'), ('QUERY_MAX_VARIANTS', '-1')])
def test_invalid_query_settings_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        if name == 'QUERY_INSTRUCTION':
            r.query_instruction(IndexConfig().embedding_model)
        else:
            r._search_queries('Question', 'Question', {})


def test_wrong_committed_version_excluded_from_search_and_terms(monkeypatch):
    c, model = setup_pipeline(monkeypatch, [record('ABC (Alpha Beta Charlie).')])
    snapshot = committed_snapshot(c)
    snapshot.owners['a']['document_version'] = 'different-write'
    monkeypatch.setattr(r, '_get_committed_snapshot', lambda c: snapshot)
    monkeypatch.setenv('QUERY_EXPANSION', 'true')
    for mode in r.RETRIEVAL_MODES:
        assert r.retrieve('ABC', retrieval_mode=mode) == []
    assert r._get_terminology(c, snapshot, r._get_bm25_index(c, snapshot)[1]) == {}


@pytest.mark.parametrize('identifier', ['RAG.v2', 'RAG/2', 'RAG+model', 'RAG#2', 'foo-RAG', 'RAG_2'])
def test_aliases_do_not_split_technical_identifiers(monkeypatch, identifier):
    monkeypatch.setenv('QUERY_EXPANSION', 'true')
    q = f'Compare {identifier} with GPT-4.'
    assert r._search_queries(q, q, {'RAG': 'Retrieval Augmented Generation'}) == [q]


def test_sentence_punctuation_allows_alias_expansion(monkeypatch):
    monkeypatch.setenv('QUERY_EXPANSION', 'true')
    q = 'Explain RAG.'
    assert r._search_queries(q, q, {'RAG': 'Retrieval Augmented Generation'})[1] == 'Explain RAG (Retrieval Augmented Generation).'
