"""Lightweight signed-cookie sessions so ONLY a verified, whitelisted @pw.live user
can reach the app's protected routes (job status, downloads) — not just /generate.

Flow: the browser signs in with Google and posts that token to /auth/session; the
server verifies it against the PW proxy allowlist ONCE and issues this HMAC-signed
session cookie (7-day expiry). Protected routes then verify the cookie locally — fast,
no proxy round-trip per request. The paid /generate action still re-checks the live
allowlist, so a user removed from the sheet mid-session is denied. The cookie holds no
secret — only the email and an expiry, signed so it cannot be forged or extended.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from . import config


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(body: str) -> str:
    mac = hmac.new(
        config.APP_SECRET_KEY.encode("utf-8"), body.encode("ascii"), hashlib.sha256
    ).digest()
    return _b64e(mac)


def make_session(email: str, *, days: int | None = None) -> str:
    """Return a signed session token carrying the email and a 7-day expiry."""
    days = config.APP_SESSION_DAYS if days is None else days
    payload = {"email": (email or "").lower(), "exp": int(time.time()) + days * 86400}
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return f"{body}.{_sign(body)}"


def verify_session(token: str) -> dict | None:
    """Return the payload if the token is validly signed and unexpired, else None.
    Fail closed: any tampering, bad signature or expiry yields None (denied)."""
    if not token or token.count(".") != 1:
        return None
    body, sig = token.split(".")
    if not hmac.compare_digest(sig, _sign(body)):
        return None
    try:
        payload = json.loads(_b64d(body))
    except Exception:  # noqa: BLE001 - malformed body
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    return payload


def email_from_google_token(token: str) -> str:
    """Best-effort read of the email claim from a Google id_token (JWT), for session
    labelling only. Returns '' if the token is not a decodable JWT. This is NOT a
    security check — the PW proxy already verified the token before we mint a session."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return ""
        data = json.loads(_b64d(parts[1]))
        return str(data.get("email", "")).lower()
    except Exception:  # noqa: BLE001
        return ""
