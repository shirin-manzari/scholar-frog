import sys
from types import SimpleNamespace

import pytest

from src.context_budget import TokenCounter, _backend_counter, generation_token_counter
from passage_helpers import CharacterTokenizer


def test_fallback_counts_utf8_bytes_and_exposes_its_limitation():
    counter = TokenCounter()
    assert counter.count('é研究') == len('é研究'.encode('utf-8'))
    assert counter.method == 'estimate:utf8-bytes'
    assert 'not a backend-token guarantee' in counter.limitation


def test_matching_local_generation_tokenizer_is_used_without_network(monkeypatch):
    calls = []
    tokenizer = CharacterTokenizer()
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda path, **kwargs: calls.append((path, kwargs)) or tokenizer)))
    _backend_counter.cache_clear()
    monkeypatch.setenv('LLM_TOKENIZER_PATH', '/local/generation-tokenizer')
    counter = generation_token_counter()
    assert counter.count('abc', special=True) == 5
    assert counter.method == 'backend-local:/local/generation-tokenizer'
    assert calls == [('/local/generation-tokenizer', {'local_files_only': True})]
    _backend_counter.cache_clear()


def test_invalid_explicit_tokenizer_path_fails_instead_of_silently_estimating(monkeypatch):
    def missing(*args, **kwargs):
        raise OSError('missing')
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=missing)))
    with pytest.raises(ValueError, match='LLM_TOKENIZER_PATH'):
        _backend_counter('ollama', 'model', '/missing/tokenizer')


def test_unavailable_backend_tokenizer_warns_once_and_never_uses_embeddings(monkeypatch):
    _backend_counter.cache_clear()
    with pytest.warns(RuntimeWarning, match='UTF-8'):
        counter = _backend_counter('anthropic', 'unknown', '')
    assert counter.tokenizer is None
    assert counter.count('abc') == 3
    _backend_counter.cache_clear()


def test_backend_chat_template_is_counted_with_generation_marker():
    calls = []
    tokenizer = SimpleNamespace(chat_template='template', apply_chat_template=lambda messages, **kwargs:
                                calls.append((messages, kwargs)) or list(range(45)))
    counter = TokenCounter(tokenizer, 'backend', '')
    assert counter.prompt_tokens('system', 'user', 32) == 77
    assert calls[0][1] == {'tokenize': True, 'add_generation_prompt': True}


def test_actual_generation_request_overflow_stops_before_backend(monkeypatch):
    from src import generate
    import src.context_budget as budgeting
    monkeypatch.setattr(budgeting, 'generation_token_counter', lambda: TokenCounter())
    monkeypatch.setenv('CONTEXT_WINDOW_TOKENS', '100')
    monkeypatch.setenv('CONTEXT_ANSWER_RESERVE', '80')
    monkeypatch.setattr(generate, '_call_ollama', lambda *args: pytest.fail('backend called'))
    with pytest.raises(ValueError, match='Generation request exceeds.*no evidence was truncated'):
        generate._call_backend('ollama', 'system', 'user')


def test_ollama_output_and_window_match_reserves(monkeypatch):
    from src import generate
    calls = []
    response = SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {'message': {'content': 'answer'}})
    monkeypatch.setattr(generate.requests, 'post', lambda *args, **kwargs: calls.append(kwargs['json']) or response)
    monkeypatch.setenv('CONTEXT_WINDOW_TOKENS', '12000')
    monkeypatch.setenv('CONTEXT_ANSWER_RESERVE', '500')
    assert generate._call_ollama('system', 'user') == 'answer'
    assert calls[0]['options'] == {'num_ctx': 12000, 'num_predict': 500}


@pytest.mark.parametrize('valid_cache', [True, False])
def test_openai_cached_encoding_is_verified_before_loading(tmp_path, monkeypatch, valid_cache):
    import hashlib
    import src.context_budget as budgeting
    table = b'cached encoding table'
    name = 'cl100k_base'
    url = f'https://openaipublic.blob.core.windows.net/encodings/{name}.tiktoken'
    (tmp_path / hashlib.sha1(url.encode()).hexdigest()).write_bytes(table if valid_cache else b'corrupt')
    monkeypatch.setenv('TIKTOKEN_CACHE_DIR', str(tmp_path))
    calls = []
    encoding = SimpleNamespace(encode=lambda text, **kwargs: list(text))
    monkeypatch.setitem(sys.modules, 'tiktoken', SimpleNamespace(
        model=SimpleNamespace(encoding_name_for_model=lambda model: name),
        encoding_for_model=lambda model: calls.append(model) or encoding))
    monkeypatch.setitem(sys.modules, 'tiktoken_ext', SimpleNamespace(
        openai_public=SimpleNamespace(cl100k_base=lambda: None)))
    monkeypatch.setattr(budgeting.inspect, 'getsource', lambda f: f'expected_hash="{hashlib.sha256(table).hexdigest()}"')
    _backend_counter.cache_clear()
    if valid_cache:
        counter = _backend_counter('openai', 'model', '')
        assert counter.method == 'backend:tiktoken:cl100k_base'
        assert counter.count('abc') == 3
        assert calls == ['model']
    else:
        with pytest.warns(RuntimeWarning):
            counter = _backend_counter('openai', 'model', '')
        assert counter.tokenizer is None and calls == []
    _backend_counter.cache_clear()
