import os

import requests
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are scholarq, an academic research assistant.
Answer only from the supplied excerpts. Every factual claim must include an
inline citation in exactly one of these forms: [Paper Title, Section, p.N]
when a section name is provided, or [Paper Title, p.N] when it is not. Use
only titles, section names, and page numbers that appear in the excerpts; never
invent a source or citation.
If the excerpts do not support an answer, say: "Not covered in the provided
excerpts." Do not use outside knowledge."""


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        section = c.get("section")
        if section and section != "Untitled section":
            citation = f"[{c['title']}, {section}, p.{c['page']}]"
        else:
            citation = f"[{c['title']}, p.{c['page']}]"
        blocks.append(f"{citation}\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    context = build_context(chunks)
    return f"""Excerpts:

{context}

---

Question: {question}

Answer (with inline citations as instructed):"""


def _call_ollama(system: str, user: str) -> str:
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "180"))
    try:
        resp = requests.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "think": False,
                "stream": False,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. Start Ollama or set OLLAMA_URL in .env."
        ) from exc
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Ollama did not respond within {timeout:g} seconds. Check that model "
            f"'{model}' is available, or increase OLLAMA_TIMEOUT in .env."
        ) from exc
    except requests.HTTPError as exc:
        details = resp.text.strip()
        if resp.status_code == 404:
            message = f"Ollama model '{model}' was not found. Install it with: ollama pull {model}"
        else:
            message = f"Ollama returned HTTP {resp.status_code}: {details or exc}"
        raise RuntimeError(message) from exc
    except (KeyError, ValueError) as exc:
        raise RuntimeError("Ollama returned an unexpected response.") from exc


def _call_openai(system: str, user: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


def _call_anthropic(system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    resp = client.messages.create(
        model=model,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def generate_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant excerpts found in the ingested papers for this question."

    user_prompt = build_user_prompt(question, chunks)
    backend = os.getenv("LLM_BACKEND", "ollama").lower()

    if backend == "ollama":
        return _call_ollama(SYSTEM_PROMPT, user_prompt)
    elif backend == "openai":
        return _call_openai(SYSTEM_PROMPT, user_prompt)
    elif backend == "anthropic":
        return _call_anthropic(SYSTEM_PROMPT, user_prompt)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend}")
