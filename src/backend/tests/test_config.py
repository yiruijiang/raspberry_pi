"""
test_config.py — Unit tests for config.py.

These tests verify that Settings loads from environment variables correctly
and that the singleton cache can be cleared between test cases.
"""

from __future__ import annotations

import os
import pytest

from backend.config import get_settings, Settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the lru_cache between tests so env var changes take effect."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults():
    settings = get_settings()
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.camera_width == 640
    assert settings.camera_height == 480
    assert settings.camera_fps == 15
    assert settings.jpeg_quality == 75
    assert settings.audio_sample_rate == 16000
    assert settings.audio_channels == 1
    assert settings.alert_buffer_size == 50
    assert settings.alert_max_age_seconds == 30
    assert settings.jwt_algorithm == "HS256"


def test_override_via_env(monkeypatch):
    monkeypatch.setenv("CAMERA_WIDTH", "1280")
    monkeypatch.setenv("CAMERA_HEIGHT", "720")
    monkeypatch.setenv("CAMERA_FPS", "30")
    monkeypatch.setenv("JPEG_QUALITY", "85")
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.camera_width == 1280
    assert settings.camera_height == 720
    assert settings.camera_fps == 30
    assert settings.jpeg_quality == 85


def test_singleton_is_cached():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_port_validation():
    with pytest.raises(Exception):
        Settings(port=0)   # below minimum

    with pytest.raises(Exception):
        Settings(port=99999)  # above maximum


def test_jwt_algorithm_validation():
    with pytest.raises(Exception):
        Settings(jwt_algorithm="HMAC512")  # not in Literal


def test_ml_socket_path_env(monkeypatch):
    monkeypatch.setenv("ML_SOCKET_PATH", "/run/baby/ml.sock")
    get_settings.cache_clear()
    assert get_settings().ml_socket_path == "/run/baby/ml.sock"
