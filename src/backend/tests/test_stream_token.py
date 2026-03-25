"""
test_stream_token.py — Unit and integration tests for TASK-105 stream token auth.

Tests cover:
  - create_stream_token / _verify_stream_token (unit)
  - POST /v1/auth/stream-token endpoint (integration)
  - /v1/stream/video and /v1/stream/audio accepting stream_token query param
  - /v1/stream/video and /v1/stream/audio still accepting Bearer JWT
  - Rejection of absent, malformed, tampered, and expired tokens
"""

from __future__ import annotations

import os
import time
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Set env before importing the app so Settings picks them up
os.environ.setdefault("JWT_SECRET", "test-secret-at-least-32-chars-long!!")
os.environ.setdefault("MONITOR_USERNAME", "testuser")
os.environ.setdefault("MONITOR_PASSWORD", "testpass")

from backend.config import get_settings
from backend.auth import create_access_token, create_stream_token, _verify_stream_token
from backend.main import app


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def bearer_headers(client):
    """Return Authorization headers from a freshly-issued JWT."""
    resp = client.post("/v1/auth/token", auth=("testuser", "testpass"))
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- #
# Unit — create_stream_token / _verify_stream_token                           #
# --------------------------------------------------------------------------- #


def test_stream_token_roundtrip():
    token = create_stream_token("alice")
    payload = _verify_stream_token(token)
    assert payload["sub"] == "alice"
    assert "iat" in payload
    assert "exp" in payload


def test_stream_token_exp_is_in_future():
    token = create_stream_token("bob")
    payload = _verify_stream_token(token)
    assert payload["exp"] > int(time.time())


def test_stream_token_missing_raises_401():
    with pytest.raises(HTTPException) as exc_info:
        _verify_stream_token("")
    assert exc_info.value.status_code == 401


def test_stream_token_no_separator_raises_401():
    with pytest.raises(HTTPException) as exc_info:
        _verify_stream_token("nodotinthisstring")
    assert exc_info.value.status_code == 401


def test_stream_token_tampered_signature_raises_401():
    token = create_stream_token("alice")
    payload_part, sig_part = token.rsplit(".", 1)
    # Flip last char of signature
    bad_sig = sig_part[:-1] + ("A" if sig_part[-1] != "A" else "B")
    with pytest.raises(HTTPException) as exc_info:
        _verify_stream_token(f"{payload_part}.{bad_sig}")
    assert exc_info.value.status_code == 401


def test_stream_token_tampered_payload_raises_401():
    token = create_stream_token("alice")
    # Replace payload with a fresh token's payload (different HMAC)
    other = create_stream_token("evil")
    other_payload, _ = other.rsplit(".", 1)
    _, original_sig = token.rsplit(".", 1)
    with pytest.raises(HTTPException) as exc_info:
        _verify_stream_token(f"{other_payload}.{original_sig}")
    assert exc_info.value.status_code == 401


def test_stream_token_expired_raises_401():
    # Issue a token and then fast-forward time past its expiry
    token = create_stream_token("alice")
    settings = get_settings()
    future_time = int(time.time()) + settings.stream_token_ttl_seconds + 10
    with patch("backend.auth.time") as mock_time:
        mock_time.time.return_value = future_time
        with pytest.raises(HTTPException) as exc_info:
            _verify_stream_token(token)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


def test_stream_token_independent_secret():
    """Tokens signed with a different secret must fail verification."""
    os.environ["STREAM_TOKEN_SECRET"] = "first-secret-value-32chars-padded!"
    get_settings.cache_clear()
    token = create_stream_token("alice")

    # Rotate to a different secret
    os.environ["STREAM_TOKEN_SECRET"] = "second-secret-value-32chars-paddd!"
    get_settings.cache_clear()
    with pytest.raises(HTTPException) as exc_info:
        _verify_stream_token(token)
    assert exc_info.value.status_code == 401

    # Cleanup
    del os.environ["STREAM_TOKEN_SECRET"]
    get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Integration — POST /v1/auth/stream-token                                    #
# --------------------------------------------------------------------------- #


def test_issue_stream_token_requires_bearer(client):
    resp = client.post("/v1/auth/stream-token")
    assert resp.status_code == 401


def test_issue_stream_token_rejects_bad_bearer(client):
    resp = client.post(
        "/v1/auth/stream-token",
        headers={"Authorization": "Bearer notavalidtoken"},
    )
    assert resp.status_code == 401


def test_issue_stream_token_success(client, bearer_headers):
    resp = client.post("/v1/auth/stream-token", headers=bearer_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "stream_token" in body
    assert body["expires_in"] == 60
    # Token must be verifiable
    payload = _verify_stream_token(body["stream_token"])
    assert payload["sub"] == "testuser"


def test_issue_stream_token_returns_correct_expires_in(client, bearer_headers):
    """expires_in must be an integer matching STREAM_TOKEN_TTL_SECONDS."""
    resp = client.post("/v1/auth/stream-token", headers=bearer_headers)
    assert resp.status_code == 200
    settings = get_settings()
    assert resp.json()["expires_in"] == settings.stream_token_ttl_seconds


# --------------------------------------------------------------------------- #
# Integration — stream endpoints accept stream_token query param              #
# --------------------------------------------------------------------------- #


def test_stream_video_rejects_no_auth(client):
    resp = client.get("/v1/stream/video")
    assert resp.status_code == 401


def test_stream_audio_rejects_no_auth(client):
    resp = client.get("/v1/stream/audio")
    assert resp.status_code == 401


def test_stream_video_rejects_invalid_stream_token(client):
    resp = client.get("/v1/stream/video?stream_token=bad.token")
    assert resp.status_code == 401


def test_stream_audio_rejects_invalid_stream_token(client):
    resp = client.get("/v1/stream/audio?stream_token=bad.token")
    assert resp.status_code == 401


def test_stream_video_accepts_bearer_jwt(client, bearer_headers):
    """Existing Bearer JWT auth must still work on /v1/stream/video."""
    # Camera is not available in CI; we expect 503, not 401/403
    resp = client.get("/v1/stream/video", headers=bearer_headers)
    assert resp.status_code in (200, 503), (
        f"Expected 200 or 503 (auth passed), got {resp.status_code}"
    )


def test_stream_audio_accepts_bearer_jwt(client, bearer_headers):
    """Existing Bearer JWT auth must still work on /v1/stream/audio."""
    resp = client.get("/v1/stream/audio", headers=bearer_headers)
    assert resp.status_code in (200, 503), (
        f"Expected 200 or 503 (auth passed), got {resp.status_code}"
    )


def test_stream_video_accepts_stream_token(client, bearer_headers):
    """stream_token query param must be accepted on /v1/stream/video."""
    token_resp = client.post("/v1/auth/stream-token", headers=bearer_headers)
    stream_token = token_resp.json()["stream_token"]

    resp = client.get(f"/v1/stream/video?stream_token={stream_token}")
    # 200 = stream started; 503 = camera not available; both mean auth passed
    assert resp.status_code in (200, 503), (
        f"Expected 200 or 503 (auth passed), got {resp.status_code}"
    )


def test_stream_audio_accepts_stream_token(client, bearer_headers):
    """stream_token query param must be accepted on /v1/stream/audio."""
    token_resp = client.post("/v1/auth/stream-token", headers=bearer_headers)
    stream_token = token_resp.json()["stream_token"]

    resp = client.get(f"/v1/stream/audio?stream_token={stream_token}")
    assert resp.status_code in (200, 503), (
        f"Expected 200 or 503 (auth passed), got {resp.status_code}"
    )


def test_stream_video_expired_stream_token_rejected(client, bearer_headers):
    """An expired stream token must produce 401 on the stream endpoint."""
    token_resp = client.post("/v1/auth/stream-token", headers=bearer_headers)
    stream_token = token_resp.json()["stream_token"]

    settings = get_settings()
    future = int(time.time()) + settings.stream_token_ttl_seconds + 10
    with patch("backend.auth.time") as mock_time:
        mock_time.time.return_value = future
        resp = client.get(f"/v1/stream/video?stream_token={stream_token}")
    assert resp.status_code == 401
