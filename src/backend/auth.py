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

Never log the raw token value.
"""

from __future__ import annotations

import logging
import os
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
