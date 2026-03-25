"""
test_auth.py — Unit tests for auth.py.

Tests cover token creation, validation, expiry, and the FastAPI dependency
helpers using the ASGI test client (httpx).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

# Patch env before importing the app so config picks it up
os.environ.setdefault("JWT_SECRET", "test-secret-at-least-32-chars-long!!")
os.environ.setdefault("MONITOR_USERNAME", "testuser")
os.environ.setdefault("MONITOR_PASSWORD", "testpass")

from backend.config import get_settings
from backend.auth import create_access_token, _decode_token
from backend.main import app


@pytest.fixture(autouse=True)
def clear_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ── Token creation ──────────────────────────────────────────────────────────

def test_create_access_token_has_required_claims():
    token = create_access_token("alice")
    settings = get_settings()
    payload = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
    )
    assert payload["sub"] == "alice"
    assert "exp" in payload
    assert "iat" in payload


def test_token_expiry_is_in_future():
    token = create_access_token("alice")
    settings = get_settings()
    payload = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
    )
    exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    assert exp > datetime.now(tz=timezone.utc)


# ── Token validation ────────────────────────────────────────────────────────

def test_valid_token_decodes():
    token = create_access_token("bob")
    payload = _decode_token(token)
    assert payload["sub"] == "bob"


def test_invalid_token_raises_401():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        _decode_token("not.a.real.token")
    assert exc_info.value.status_code == 401


def test_tampered_token_raises_401():
    from fastapi import HTTPException
    token = create_access_token("alice")
    tampered = token[:-4] + "XXXX"
    with pytest.raises(HTTPException):
        _decode_token(tampered)


# ── /v1/auth/token endpoint ─────────────────────────────────────────────────

def test_auth_token_endpoint_success(client):
    response = client.post(
        "/v1/auth/token",
        auth=("testuser", "testpass"),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert "access_token" in body
    assert "expires_in" in body


def test_auth_token_endpoint_bad_password(client):
    response = client.post(
        "/v1/auth/token",
        auth=("testuser", "wrongpassword"),
    )
    assert response.status_code == 401


def test_auth_token_endpoint_no_credentials(client):
    response = client.post("/v1/auth/token")
    assert response.status_code == 401


# ── Protected endpoints require JWT ─────────────────────────────────────────

def test_stream_status_requires_auth(client):
    response = client.get("/v1/stream/status")
    assert response.status_code == 401


def test_stream_status_with_valid_token(client):
    # Get a token first
    token_resp = client.post("/v1/auth/token", auth=("testuser", "testpass"))
    token = token_resp.json()["access_token"]

    response = client.get(
        "/v1/stream/status",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "camera" in body
    assert "audio" in body
    assert "ws_clients" in body


def test_recent_alerts_requires_auth(client):
    response = client.get("/v1/alerts/recent")
    assert response.status_code == 401


def test_healthz_requires_no_auth(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
