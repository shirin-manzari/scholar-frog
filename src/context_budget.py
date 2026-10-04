"""Offline generation token accounting; never uses the embedding tokenizer."""
import hashlib
import inspect
import re
import os
import tempfile
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def positive_setting(name, default):
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


@dataclass(frozen=True)
class TokenCounter:
    tokenizer: object = None
    method: str = 'estimate:utf8-bytes'
    limitation: str = 'One token per UTF-8 byte is a conservative estimate, not a backend-token guarantee.'

    def count(self, text, *, special=False):
        if self.tokenizer is None:
            return len(text.encode('utf-8'))
        return len(self.tokenizer.encode(text, add_special_tokens=special, truncation=False))

    def prompt_tokens(self, system, user, framing):
        if self.tokenizer is not None and getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            return len(self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True)) + framing
        return self.count(system, special=True) + self.count(user, special=True) + framing


class _TikToken:
    def __init__(self, encoding):
        self.encoding = encoding

    def encode(self, text, **kwargs):
        return self.encoding.encode(text, disallowed_special=())


@lru_cache(maxsize=8)
def _backend_counter(backend, model, path):
    if path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        except Exception as exc:
            raise ValueError('LLM_TOKENIZER_PATH must identify a locally available tokenizer matching '
                             'the generation model; no download is attempted.') from exc
        return TokenCounter(tokenizer, f'backend-local:{path}', '')
    if backend == 'openai':
        try:
            import tiktoken
            name = tiktoken.model.encoding_name_for_model(model)
            # tiktoken can download tables on first use. Only load a cached table.
            url = f'https://openaipublic.blob.core.windows.net/encodings/{name}.tiktoken'
            cache = Path(os.getenv('TIKTOKEN_CACHE_DIR', os.getenv('DATA_GYM_CACHE_DIR',
                         str(Path(tempfile.gettempdir()) / 'data-gym-cache'))))
            if name in {'cl100k_base', 'o200k_base'}:
                from tiktoken_ext import openai_public
                # Verify the package's expected table digest before its loader runs.
                # Missing/corrupt caches must never trigger an implicit download.
                constructor = inspect.getsource(getattr(openai_public, name))
                expected = re.search(r'expected_hash=["\']([0-9a-f]{64})["\']', constructor)
                table = cache / hashlib.sha1(url.encode()).hexdigest()
                if expected and table.is_file() and hashlib.sha256(table.read_bytes()).hexdigest() == expected[1]:
                    return TokenCounter(_TikToken(tiktoken.encoding_for_model(model)), f'backend:tiktoken:{name}', '')
        except (ImportError, KeyError, OSError, TypeError, AttributeError):
            pass
    warnings.warn('Generation tokenizer unavailable locally; context budgets use one token per UTF-8 '
                  'byte. Set LLM_TOKENIZER_PATH to a matching local tokenizer for tighter accounting.',
                  RuntimeWarning, stacklevel=2)
    return TokenCounter()


def generation_token_counter():
    backend = os.getenv('LLM_BACKEND', 'ollama').lower()
    model = os.getenv(f'{backend.upper()}_MODEL', '')
    return _backend_counter(backend, model, os.getenv('LLM_TOKENIZER_PATH', '').strip())
