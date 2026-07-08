"""Thin wrapper around the google-genai SDK: embeddings, generation, file parsing."""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from . import config

_EMBED_BATCH = 100
_MAX_RETRIES = 4


class GeminiJSONError(ValueError):
    """Raised when Gemini returns text that cannot be parsed as JSON."""


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    return genai.Client(api_key=config.require_api_key())


def _retry(fn, *args, **kwargs):
    """Call `fn` with naive exponential backoff on transient API errors."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - SDK raises a variety of errors
            msg = str(exc).lower()
            transient = any(t in msg for t in ("429", "503", "500", "deadline", "timeout", "unavailable"))
            if not transient or attempt == _MAX_RETRIES - 1:
                raise
            last_exc = exc
            time.sleep(2 ** attempt)
    if last_exc:
        raise last_exc


# --- Embeddings -----------------------------------------------------------
def embed_texts(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    """Return one embedding vector per input text. Batched to respect API limits."""
    if not texts:
        return []
    client = get_client()
    task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
    out: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = texts[start : start + _EMBED_BATCH]
        resp = _retry(
            client.models.embed_content,
            model=config.GEMINI_EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=config.EMBED_DIM
            ),
        )
        out.extend(e.values for e in resp.embeddings)
    return out


# --- Generation -----------------------------------------------------------
def image_part(data: bytes, *, mime_type: str = "image/png"):
    """Wrap raw image bytes as a content Part so it can be passed in `attachments`
    alongside (or instead of) uploaded files."""
    return types.Part.from_bytes(data=data, mime_type=mime_type)


def generate_text(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.3,
    attachments: list | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """Generate text. `attachments` (e.g. an uploaded PDF) are prepended so the
    model can see figures/diagrams alongside the prompt."""
    client = get_client()
    contents = prompt if not attachments else [*attachments, prompt]
    resp = _retry(
        client.models.generate_content,
        model=model or config.GEMINI_GEN_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system,
            max_output_tokens=max_output_tokens,
        ),
    )
    return (resp.text or "").strip()


def _loads_json(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.lstrip().lower().startswith("json"):
            raw = raw.lstrip()[4:]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


def _json_retry_contents(contents: Any, exc: json.JSONDecodeError) -> Any:
    hint = (
        "\n\nIMPORTANT: The previous response was not valid JSON "
        f"({exc.msg} at line {exc.lineno}, column {exc.colno}). "
        "Regenerate the COMPLETE result as exactly one valid JSON value. "
        "Escape quotes and newlines inside strings, close every string/object/array, "
        "and output no commentary."
    )
    if isinstance(contents, str):
        return contents + hint
    if isinstance(contents, list):
        return [*contents, hint]
    return [contents, hint]


def generate_json(
    contents: Any,
    *,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.1,
    max_output_tokens: int | None = None,
) -> Any:
    """Generate and parse a JSON response. `contents` may include uploaded files."""
    client = get_client()
    token_limit = (
        config.GEMINI_MAX_OUTPUT_TOKENS
        if max_output_tokens is None
        else max_output_tokens
    )
    current_contents = contents
    last_exc: json.JSONDecodeError | None = None
    for attempt in range(2):
        resp = _retry(
            client.models.generate_content,
            model=model or config.GEMINI_PARSE_MODEL,
            contents=current_contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system,
                response_mime_type="application/json",
                max_output_tokens=token_limit,
            ),
        )
        try:
            return _loads_json(resp.text or "{}")
        except json.JSONDecodeError as exc:
            last_exc = exc
            if attempt == 0:
                current_contents = _json_retry_contents(contents, exc)
                continue
    assert last_exc is not None
    raise GeminiJSONError(
        "Gemini returned incomplete or malformed JSON "
        f"({last_exc.msg} at line {last_exc.lineno}, column {last_exc.colno})."
    ) from last_exc


# --- File upload (for multimodal parsing) ---------------------------------
def upload_file(path: str | Path):
    """Upload a file via the Gemini File API and wait until it is ACTIVE."""
    client = get_client()
    f = _retry(client.files.upload, file=str(path))
    # Files become processable after a short server-side step.
    for _ in range(30):
        if getattr(f.state, "name", str(f.state)) == "ACTIVE":
            return f
        time.sleep(1)
        f = client.files.get(name=f.name)
    return f
