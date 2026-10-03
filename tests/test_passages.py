"""Token passage and committed expansion contracts, independent of model downloads."""
import json
from types import SimpleNamespace

import pytest

from passage_helpers import CharacterTokenizer
from src import ingest, retrieve, sync
from src.citations import assign_evidence
from src.generate import build_context, build_user_prompt, SYSTEM_PROMPT
from src.index_config import IndexConfig, IndexCompatibilityError, check_compatibility, new_metadata
from src.sync import CommittedSnapshot

TOKENIZER = CharacterTokenizer()


def spans(text, size=60, overlap=15):
    paragraphs, _, _ = ingest.page_paragraphs(text)
    return ingest.passage_spans(text, paragraphs, size=size, overlap=overlap,
                                tokenizer=TOKENIZER, max_tokens=510)


def test_paragraphs_sentences_and_exact_locations():
    text = "# Methods\n\nA short paragraph.\n\nFirst full sentence. Second full sentence. Third full sentence."
    passages = spans(text, size=45, overlap=22)
    chunks = [text[p['start']:p['end']] for p in passages]
    assert any('A short paragraph.' in chunk for chunk in chunks)
    for sentence in ('First full sentence.', 'Second full sentence.', 'Third full sentence.'):
        assert any(sentence in chunk for chunk in chunks)
        assert all(not chunk.endswith(sentence[:-1]) for chunk in chunks)
    assert all(p['section'] == 'Methods' for p in passages)
    assert all(len(chunk) <= 45 for chunk in chunks)
    covered = {i for p in passages for i in range(p['start'], p['end'])}
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())


def test_oversized_sentence_overlap_without_missing_characters():
    text = 'abcdefghijklmnopqrstuvwxyz' * 4
    passages = spans(text, size=20, overlap=5)
    assert all(p['end'] - p['start'] <= 20 for p in passages)
    assert all(left['end'] - right['start'] == 5 for left, right in zip(passages, passages[1:]))
    assert set(range(len(text))) == {i for p in passages for i in range(p['start'], p['end'])}


def test_tokenizer_capacity_includes_special_tokens_and_model_limit():
    tokenizer, limit = ingest.tokenizer_limits(SimpleNamespace(tokenizer=TOKENIZER, max_seq_length=32))
    assert limit == 30
    text = 'a' * 100
    paragraphs, _, _ = ingest.page_paragraphs(text)
    for passage in ingest.passage_spans(text, paragraphs, size=300, overlap=50,
                                      tokenizer=tokenizer, max_tokens=limit):
        assert ingest.token_count(text[passage['start']:passage['end']], tokenizer, special=True) <= 32


def test_sentence_overlap_preserves_whole_sentences():
    text = 'First sentence. Second sentence. Third sentence. Fourth sentence.'
    passages = spans(text, size=40, overlap=20)
    chunks = [text[p['start']:p['end']] for p in passages]
    assert chunks[0] == 'First sentence. Second sentence.'
    assert chunks[1].startswith('Second sentence.')
    assert all(chunk.endswith('.') for chunk in chunks)


def corpus_for(pages, doc='document', version='write1'):
    corpus = []
    section, section_id = 'Untitled section', 0
    for page, text in pages:
        paragraphs, section, section_id = ingest.page_paragraphs(text, section, section_id)
        for index, para in enumerate(paragraphs):
            metadata = {**para, 'source': doc + '.pdf', 'title': 'Paper', 'page': page,
                        'document_id': doc, 'file_hash': doc, 'document_version': version,
                        'character_start': para['start'], 'character_end': para['end'],
                        'paragraph_start': index, 'paragraph_end': index,
                        'page_context': json.dumps({'text': text, 'paragraphs': paragraphs})}
            corpus.append(retrieve._make_result(f'{doc}-{version}-{page}-{index}',
                                               text[para['start']:para['end']], metadata))
    return corpus


def snapshot_for(corpus):
    owners = {hit['id']: {'document_id': hit['metadata']['document_id'], 'paths': [hit['source']]}
              for hit in corpus}
    return CommittedSnapshot('test', frozenset(owners), owners)


def expand(anchors, corpus, **kwargs):
    return retrieve.expand_context('Question?', anchors, corpus, snapshot_for(corpus),
                                  tokenizer=TOKENIZER, budget=10000, window=20000,
                                  answer_reserve=500, **kwargs)


def test_expansion_section_document_version_and_page_boundaries():
    corpus = corpus_for([(1, '# Methods\n\nBefore.\n\nAnchor.'),
                         (2, 'Continuation.\n\n## Results\n\nUnrelated findings.')])
    foreign = corpus_for([(1, '# Methods\n\nFOREIGN.')], doc='foreign')
    old = corpus_for([(1, '# Methods\n\nOLD VERSION.')], version='write0')
    anchor = next(hit for hit in corpus if hit['text'] == 'Anchor.')
    anchor['distance'] = 0.123
    results = expand([anchor], corpus + foreign + old, radius=5)
    assert [result['page'] for result in results] == [1, 2]
    assert results[0]['text'] == '# Methods\n\nBefore.\n\nAnchor.'
    assert results[1]['text'] == 'Continuation.'
    assert all(result['distance'] == 0.123 for result in results)
    assert all(result['metadata']['document_version'] == 'write1' for result in results)
    for result in results:
        page = next(text for number, text in [(1, '# Methods\n\nBefore.\n\nAnchor.'),
                                              (2, 'Continuation.\n\n## Results\n\nUnrelated findings.')]
                    if number == result['page'])
        meta = result['metadata']
        assert page[meta['character_start']:meta['character_end']] == result['text']


def test_overlap_deduplication_preserves_all_anchors_and_their_scores():
    corpus = corpus_for([(1, '# Methods\n\nOne.\n\nTwo.\n\nThree.')])
    anchors = corpus[1:3]
    anchors[0]['distance'], anchors[1]['distance'] = 0.1, 0.2
    results = expand(anchors, corpus, radius=2)
    assert len(results) == 1
    assert results[0]['text'].count('Two.') == 1
    assert [hit['id'] for hit in results[0]['anchor_hits']] == [hit['id'] for hit in anchors]
    assert [hit['scores']['distance'] for hit in results[0]['anchor_hits']] == [0.1, 0.2]


def test_budget_enforcement_and_mandatory_anchor_retention():
    corpus = corpus_for([(1, '# Methods\n\nBefore.' + 'x' * 300 + '\n\nAnchor.\n\nAfter.' + 'y' * 300)])
    anchor = corpus[2]
    base = expand([anchor], corpus, radius=0)
    budget = ingest.token_count(build_context(base), TOKENIZER)
    result = retrieve.expand_context('Question?', [anchor], corpus, snapshot_for(corpus),
                                    tokenizer=TOKENIZER, radius=2, budget=budget,
                                    window=20000, answer_reserve=500)
    assert result[0]['text'] == 'Anchor.'
    assert ingest.token_count(build_context(result), TOKENIZER) <= budget
    prompt_size = ingest.token_count(SYSTEM_PROMPT + '\n' + build_user_prompt('Question?', result), TOKENIZER, special=True)
    with pytest.raises(ValueError, match='no evidence was truncated'):
        retrieve.expand_context('Question?', [anchor], corpus, snapshot_for(corpus),
                                tokenizer=TOKENIZER, radius=1, budget=budget,
                                window=prompt_size + 499, answer_reserve=500)


def test_pending_records_excluded_and_uncommitted_anchor_rejected():
    committed = corpus_for([(1, '# Methods\n\nAnchor.')])
    pending = corpus_for([(2, 'PENDING.')])
    snapshot = snapshot_for(committed)
    result = retrieve.expand_context('Question?', [committed[1]], committed + pending, snapshot,
                                     tokenizer=TOKENIZER, radius=10, budget=10000,
                                     window=20000, answer_reserve=500)
    assert all('PENDING' not in item['text'] for item in result)
    with pytest.raises(ValueError, match='uncommitted'):
        retrieve.expand_context('Question?', pending, committed + pending, snapshot,
                                tokenizer=TOKENIZER, budget=10000, window=20000, answer_reserve=500)


def test_evidence_ids_and_excerpts_map_to_exact_expanded_page_text():
    corpus = corpus_for([(1, '# Methods\n\nAnchor.'), (2, 'Continuation.')])
    expanded = expand([corpus[1]], corpus, radius=3)
    evidence = assign_evidence(expanded)
    assert len(evidence) == 2
    prompt = build_context(expanded)
    for item, passage in zip(evidence, expanded):
        assert item.text == passage['text']
        assert item.metadata['character_start'] == passage['metadata']['character_start']
        assert f'[{item.evidence_id}]' in prompt
        assert f'Page: {item.page}\nContent: {item.text}' in prompt


def test_old_schema_and_old_context_require_explicit_rebuild():
    config = IndexConfig()
    with pytest.raises(IndexCompatibilityError, match='python ask.py index rebuild'):
        check_compatibility(config, {**new_metadata(config), 'schema_version': 1})
    from dataclasses import replace
    with pytest.raises(IndexCompatibilityError, match='rebuild required'):
        check_compatibility(config, new_metadata(replace(config, context_version='older')))


def test_overlap_uses_trailing_sentences_of_an_intact_paragraph():
    text = 'First sentence. Second sentence.\n\nThird sentence. Fourth sentence.'
    passages = spans(text, size=55, overlap=20)
    chunks = [text[p['start']:p['end']] for p in passages]
    assert chunks[0] == 'First sentence. Second sentence.'
    assert chunks[1] == 'Second sentence.\n\nThird sentence. Fourth sentence.'


def test_repeated_section_names_do_not_expand_into_a_later_section():
    corpus = corpus_for([(1, '# Methods\n\nAnchor.\n\n# Results\n\nFinding.\n\n# Methods\n\nOther method.')])
    result = expand([corpus[1]], corpus, radius=10)
    assert len(result) == 1
    assert result[0]['text'] == '# Methods\n\nAnchor.'


def test_adjacent_selected_sections_retain_their_section_metadata():
    corpus = corpus_for([(1, '# Methods\n\nMethod.\n\n# Results\n\nResult.')])
    result = expand([corpus[1], corpus[3]], corpus, radius=1)
    assert [hit['section'] for hit in result] == ['Methods', 'Results']


def test_missing_versioned_context_fails_closed():
    corpus = corpus_for([(1, 'Anchor.')])
    corpus[0]['metadata'].pop('page_context')
    with pytest.raises(ValueError, match='rebuild'):
        expand(corpus, corpus)


def test_generation_receives_and_validates_the_expanded_evidence(monkeypatch):
    from src import generate
    corpus = corpus_for([(1, '# Methods\n\nAnchor.'), (2, 'Continuation.')])
    expanded = expand([corpus[1]], corpus, radius=3)
    prompts = []

    def backend(name, system, user):
        prompts.append(user)
        return json.dumps({'status': 'answered', 'answer': 'A continuation is reported [E2].'})

    monkeypatch.setattr(generate, '_call_backend', backend)
    result = generate.generate_answer('Question?', expanded, semantic_validation_enabled=False)
    assert result.validation.references_valid
    assert result.validation.valid_evidence[0].page == 2
    assert result.validation.valid_evidence[0].text == 'Continuation.'
    assert 'Page: 2\nContent: Continuation.' in prompts[0]
    assert result.evidence[0].text == expanded[0]['text']


def test_default_token_configuration_keeps_the_existing_embedding_model():
    config = IndexConfig()
    assert (config.chunk_size, config.chunk_overlap) == (300, 50)
    assert config.embedding_model == 'BAAI/bge-small-en-v1.5'
