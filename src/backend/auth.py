"""
auth.py — JWT authentication middleware and dependency.

All HTTP endpoints and WebSocket upgrades require a valid JWT per the
security-first design principle.  For WebSocket connections, the token is
passed as a query parameter (?token=...) because browser WebSocket APIs do
not support custom headers.

Token issuance (/v1/auth/token) uses HTTP Basic credentials checked against
environment-variable-configured credentials.  This is suitable for a single-
household deployment; for multi-user, swap in a proper user store.

Algorithm: HS256 (default).  Switch to RS256 by setting JWT_ALGORITHM=RS256
and providing JWT_SECRET as the PEM-encoded private key path.

Stream tokens (ADR-001)
-----------------------
Browsers cannot attach Authorization headers to <img src="..."> requests.
POST /v1/auth/stream-token issues a short-lived HMAC-SHA256 signed token
(default 60 s TTL) that the client appends as ?stream_token=<signed>.
The stream token is NOT a JWT — it is a compact signed payload:

    base64url(json_payload) + "." + base64url(hmac_sha256_signature)

Payload fields: sub (username), iat (issued-at Unix timestamp), exp (expiry).

The HMAC key defaults to jwt_secret but can be rotated independently via
the STREAM_TOKEN_SECRET environment variable.

Never log the raw token value.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, Header, HTTPException, Query, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from jwt import InvalidTokenError

from .config import get_settings

logger = logging.getLogger(__name__)
security = HTTPBasic()

# --------------------------------------------------------------------------- #
# Token creation                                                               #
# --------------------------------------------------------------------------- #


def create_access_token(subject: str) -> str:
    """
    Issue a signed JWT for the given subject (username).

    The token carries:
      sub  — subject (username)
      iat  — issued-at (UTC)
      exp  — expiry (UTC, settings.jwt_expire_minutes from now)
    """
    settings = get_settings()
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


# --------------------------------------------------------------------------- #
# Token validation                                                             #
# --------------------------------------------------------------------------- #


def _decode_token(token: str) -> dict:
    """
    Validate and decode a JWT string.

    Raises HTTPException(401) on any validation failure.
    Never logs the raw token value.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "exp", "iat"]},
        )
        return payload
    except InvalidTokenError as exc:
        # Log the error class but NOT the token itself
        logger.warning("JWT validation failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# --------------------------------------------------------------------------- #
# FastAPI dependency — HTTP Bearer                                             #
# --------------------------------------------------------------------------- #


async def require_jwt(
    authorization: Annotated[str | None, Header()] = None,
) -> dict:
    """
    FastAPI dependency that extracts and validates a Bearer token from the
    Authorization header.

    Usage:
        @app.get("/v1/stream/status")
        async def status(claims: Annotated[dict, Depends(require_jwt)]):
            ...
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing or malformed. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization[len("Bearer "):]
    return _decode_token(token)


# --------------------------------------------------------------------------- #
# FastAPI dependency — WebSocket query-param token                             #
# --------------------------------------------------------------------------- #


async def require_jwt_ws(
    token: Annotated[str | None, Query()] = None,
) -> dict:
    """
    FastAPI dependency for WebSocket endpoints.  The client passes the token
    as a URL query parameter: wss://host/ws/alerts?token=<jwt>

    Usage:
        @app.websocket("/ws/alerts")
        async def alerts_ws(
            websocket: WebSocket,
            claims: Annotated[dict, Depends(require_jwt_ws)],
        ):
            ...
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing token query parameter",
        )
    return _decode_token(token)


# --------------------------------------------------------------------------- #
# Stream token — HMAC-SHA256 signed, short-lived (ADR-001)                   #
# --------------------------------------------------------------------------- #

# Compact token format:  <b64url(payload_json)>.<b64url(hmac_sha256)>
# This is intentionally NOT a JWT to keep it distinct from the Bearer token
# and to allow independent secret rotation.

_STREAM_TOKEN_SEPARATOR = "."


def _stream_token_secret() -> bytes:
    """
    Return the HMAC key for stream token signing.

    Falls back to jwt_secret if STREAM_TOKEN_SECRET is not configured.
    Never logged.
    """
    settings = get_settings()
    secret = settings.stream_token_secret or settings.jwt_secret
    return secret.encode("utf-8")


def create_stream_token(subject: str) -> str:
    """
    Issue a short-lived HMAC-SHA256 stream token for the given subject.

    The token is a two-part dot-separated string:
        base64url(payload_json) + "." + base64url(hmac_sha256_signature)

    where payload_json is:
        {"sub": "<username>", "iat": <unix_ts>, "exp": <unix_ts>}

    TTL is controlled by settings.stream_token_ttl_seconds (default 60 s).
    """
    settings = get_settings()
    now = int(time.time())
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + settings.stream_token_ttl_seconds,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = urlsafe_b64encode(payload_bytes).rstrip(b"=").decode("ascii")

    sig = hmac.new(_stream_token_secret(), payload_b64.encode("ascii"), hashlib.sha256).digest()
    sig_b64 = urlsafe_b64encode(sig).rstrip(b"=").decode("ascii")

    return f"{payload_b64}{_STREAM_TOKEN_SEPARATOR}{sig_b64}"


def _verify_stream_token(token: str) -> dict:
    """
    Validate an HMAC-SHA256 stream token.

    Returns the decoded payload dict on success.
    Raises HTTPException(401) if the token is absent, malformed, has an
    invalid signature, or has expired.

    Never logs the raw token value.
    """
    if not token or _STREAM_TOKEN_SEPARATOR not in token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Stream token missing or malformed",
        )

    try:
        payload_b64, sig_b64 = token.rsplit(_STREAM_TOKEN_SEPARATOR, 1)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Stream token malformed",
        )

    # Constant-time HMAC verification
    expected_sig = hmac.new(
        _stream_token_secret(), payload_b64.encode("ascii"), hashlib.sha256
    ).digest()

    # Pad base64url string to a multiple of 4 before decoding
    def _b64pad(s: str) -> str:
        return s + "=" * (-len(s) % 4)

    try:
        provided_sig = urlsafe_b64decode(_b64pad(sig_b64))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Stream token malformed",
        )

    if not hmac.compare_digest(expected_sig, provided_sig):
        logger.warning("Stream token: HMAC verification failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Stream token invalid",
        )

    # Decode payload
    try:
        payload_bytes = urlsafe_b64decode(_b64pad(payload_b64))
        payload = json.loads(payload_bytes)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Stream token payload unreadable",
        )

    # Expiry check
    now = int(time.time())
    if payload.get("exp", 0) < now:
        logger.warning("Stream token: expired (sub=%s)", payload.get("sub", "?"))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Stream token expired",
        )

    return payload


async def require_stream_auth(
    authorization: Annotated[str | None, Header()] = None,
    stream_token: Annotated[str | None, Query()] = None,
) -> dict:
    """
    FastAPI dependency for stream endpoints (/v1/stream/video, /v1/stream/audio).

    Accepts either:
      - Authorization: Bearer <jwt>  (existing auth — not broken)
      - ?stream_token=<signed>       (new short-lived stream token, ADR-001)

    stream_token is checked first because it is the expected path for browser
    <img src="..."> and <audio src="..."> usage.  Bearer JWT is kept as a
    valid alternative so existing integrations and curl/test usage continue
    to work without modification.

    Returns the decoded claims dict (from whichever credential was provided).
    Raises HTTPException(401) if neither credential is present or valid.
    """
    if stream_token:
        return _verify_stream_token(stream_token)

    if authorization and authorization.startswith("Bearer "):
        token = authorization[len("Bearer "):]
        return _decode_token(token)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=(
            "Authentication required. "
            "Provide Authorization: Bearer <jwt> header "
            "or ?stream_token=<signed> query parameter."
        ),
        headers={"WWW-Authenticate": "Bearer"},
    )


# --------------------------------------------------------------------------- #
# Credential checking for /v1/auth/token                                      #
# --------------------------------------------------------------------------- #


def _get_admin_credentials() -> tuple[str, str]:
    """
    Read MONITOR_USERNAME / MONITOR_PASSWORD from environment.

    These are the credentials used to obtain a JWT.  They must be set via
    environment variables — never hardcoded defaults in production.
    """
    username = os.environ.get("MONITOR_USERNAME", "admin")
    password = os.environ.get("MONITOR_PASSWORD", "")
    if not password:
        logger.warning(
            "MONITOR_PASSWORD is not set. Set it via environment variable before deploying."
        )
    return username, password


def verify_basic_credentials(credentials: HTTPBasicCredentials) -> str:
    """
    Verify HTTP Basic credentials using constant-time comparison.

    Returns the username on success.
    Raises HTTPException(401) on failure.
    """
    import hmac

    expected_username, expected_password = _get_admin_credentials()

    username_match = hmac.compare_digest(
        credentials.username.encode("utf-8"),
        expected_username.encode("utf-8"),
    )
    password_match = hmac.compare_digest(
        credentials.password.encode("utf-8"),
        expected_password.encode("utf-8"),
    )

    if not (username_match and password_match):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username
