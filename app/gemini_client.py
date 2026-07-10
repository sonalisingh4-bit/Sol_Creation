"""Proxy-only Gemini helpers for generation, multimodal parts, and local search vectors."""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pw_access

from . import config

_MAX_RETRIES = 4
_TOKEN_RE = re.compile(r"[^\W\d_]{2,}|\d+", re.UNICODE)
_CTX = threading.local()


class GeminiJSONError(ValueError):
    """Raised when Gemini returns text that cannot be parsed as JSON."""


def set_proxy_context(google_token: str, session: pw_access.UsageSession | None = None) -> None:
    _CTX.google_token = google_token
    _CTX.session = session


def clear_proxy_context() -> None:
    _CTX.google_token = ""
    _CTX.session = None


@contextmanager
def proxy_context(
    google_token: str, session: pw_access.UsageSession | None = None
) -> Iterator[None]:
    previous = (getattr(_CTX, "google_token", ""), getattr(_CTX, "session", None))
    set_proxy_context(google_token, session)
    try:
        yield
    finally:
        _CTX.google_token, _CTX.session = previous


def _token() -> str:
    token = getattr(_CTX, "google_token", "")
    if not token:
        raise PermissionError("A signed-in Google token is required for Gemini calls.")
    return token


def _retry(fn, *args, **kwargs):
    """Call `fn` with naive exponential backoff on transient proxy/API errors."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - proxy/requests errors vary
            msg = str(exc).lower()
            transient = any(
                t in msg
                for t in ("429", "500", "502", "503", "504", "deadline", "timeout", "unavailable")
            )
            if not transient or attempt == _MAX_RETRIES - 1:
                raise
            last_exc = exc
            time.sleep(2 ** attempt)
    if last_exc:
        raise last_exc


# --- Local lexical vectors ------------------------------------------------
def _hash_embedding(text: str) -> list[float]:
    """Deterministic non-AI vector used only for local KB indexing/search."""
    dim = max(int(config.EMBED_DIM), 1)
    vec = [0.0] * dim
    for token in _TOKEN_RE.findall((text or "").lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[bucket] += sign
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def embed_texts(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    """Return local lexical vectors without calling an AI provider."""
    return [_hash_embedding(t) for t in texts]


# --- Multimodal parts -----------------------------------------------------
def _inline_part(data: bytes, *, mime_type: str) -> dict[str, Any]:
    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": base64.b64encode(data).decode("ascii"),
        }
    }


def image_part(data: bytes, *, mime_type: str = "image/png") -> dict[str, Any]:
    """Wrap raw image bytes as an inline Gemini content part."""
    return _inline_part(data, mime_type=mime_type)


def upload_file(path: str | Path) -> dict[str, Any]:
    """Return an inline file part; the app no longer calls Gemini's File API."""
    path = Path(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return _inline_part(path.read_bytes(), mime_type=mime_type)


# --- Generation -----------------------------------------------------------
def _as_part(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"text": item}
    if isinstance(item, dict):
        return item
    raise TypeError(f"Unsupported Gemini content part: {type(item).__name__}")


def _contents(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list) and all(isinstance(v, dict) and "parts" in v for v in value):
        return value
    if isinstance(value, dict) and "parts" in value:
        return [value]
    items = value if isinstance(value, list) else [value]
    return [{"role": "user", "parts": [_as_part(item) for item in items if item is not None]}]


def _request(
    contents: Any,
    *,
    system: str | None,
    temperature: float,
    max_output_tokens: int | None,
    response_mime_type: str | None = None,
) -> dict[str, Any]:
    generation_config: dict[str, Any] = {"temperature": temperature}
    if max_output_tokens is not None:
        generation_config["maxOutputTokens"] = max_output_tokens
    if response_mime_type:
        generation_config["responseMimeType"] = response_mime_type
    req: dict[str, Any] = {
        "contents": _contents(contents),
        "generationConfig": generation_config,
    }
    if system:
        req["systemInstruction"] = {"parts": [{"text": system}]}
    return req


def _text_from_result(result: Any) -> str:
    if isinstance(result, dict):
        if isinstance(result.get("text"), str):
            return result["text"]
        candidates = result.get("candidates") or []
        pieces: list[str] = []
        for candidate in candidates:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    pieces.append(part["text"])
        return "".join(pieces)
    return ""


def _proxy_generate(
    *,
    model: str,
    request: dict[str, Any],
    filename: str = "",
    input_unit: str = "",
    count: Any = None,
) -> dict[str, Any]:
    return _retry(
        pw_access.gemini_generate,
        _token(),
        model=model,
        request=request,
        filename=filename,
        input_unit=input_unit,
        count=count,
        session=getattr(_CTX, "session", None),
    )


def generate_text(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.3,
    attachments: list | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """Generate text through the PW proxy."""
    contents = prompt if not attachments else [*attachments, prompt]
    req = _request(
        contents,
        system=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    resp = _proxy_generate(model=model or config.GEMINI_GEN_MODEL, request=req)
    return _text_from_result(resp.get("result")).strip()


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
    """Generate and parse a JSON response through the PW proxy."""
    token_limit = (
        config.GEMINI_MAX_OUTPUT_TOKENS
        if max_output_tokens is None
        else max_output_tokens
    )
    current_contents = contents
    last_exc: json.JSONDecodeError | None = None
    for attempt in range(2):
        req = _request(
            current_contents,
            system=system,
            temperature=temperature,
            response_mime_type="application/json",
            max_output_tokens=token_limit,
        )
        resp = _proxy_generate(model=model or config.GEMINI_PARSE_MODEL, request=req)
        raw = _text_from_result(resp.get("result")) or "{}"
        try:
            return _loads_json(raw)
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
