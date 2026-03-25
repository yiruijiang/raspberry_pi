"""
config.py — Centralised configuration for the Baby Monitor backend.

All tuneable values are read from environment variables (with sensible
defaults), so nothing security-sensitive is hardcoded and the same image
runs on the Pi with a .env file or in CI with injected variables.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings.  Loaded once at startup via get_settings().

    All fields can be overridden with environment variables of the same name
    (case-insensitive).  A .env file in the working directory is also loaded.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Server                                                               #
    # ------------------------------------------------------------------ #
    host: str = Field(default="0.0.0.0", description="Interface to bind on.")
    port: int = Field(default=8000, ge=1, le=65535, description="TCP port.")
    debug: bool = Field(default=False, description="Enable Uvicorn debug/reload.")

    # ------------------------------------------------------------------ #
    # TLS (production: point at real cert/key files)                       #
    # ------------------------------------------------------------------ #
    tls_certfile: str | None = Field(
        default=None,
        description="Path to PEM certificate for HTTPS/WSS. None → plain HTTP (dev only).",
    )
    tls_keyfile: str | None = Field(
        default=None,
        description="Path to PEM private key for HTTPS/WSS.",
    )

    # ------------------------------------------------------------------ #
    # JWT authentication                                                   #
    # ------------------------------------------------------------------ #
    jwt_secret: str = Field(
        default="CHANGE_ME_IN_PRODUCTION",
        description=(
            "HS256 signing secret (min 32 chars recommended). "
            "In production, set via environment variable — never commit the real value."
        ),
    )
    jwt_algorithm: Literal["HS256", "RS256"] = Field(
        default="HS256",
        description="JWT signing algorithm. Switch to RS256 for asymmetric keys.",
    )
    jwt_expire_minutes: int = Field(
        default=720,  # 12 hours
        ge=1,
        description="Token validity window in minutes.",
    )

    # ------------------------------------------------------------------ #
    # Camera / video                                                       #
    # ------------------------------------------------------------------ #
    camera_index: int = Field(
        default=0,
        ge=0,
        description=(
            "libcamera / V4L2 camera index. "
            "0 is the CSI camera on a Pi with a single attached module."
        ),
    )
    camera_width: int = Field(default=640, ge=160, description="Capture width in pixels.")
    camera_height: int = Field(default=480, ge=120, description="Capture height in pixels.")
    camera_fps: int = Field(
        default=15,
        ge=1,
        le=60,
        description="Target capture frame rate. 15 fps is spec minimum.",
    )
    jpeg_quality: int = Field(
        default=75,
        ge=1,
        le=95,
        description="JPEG encode quality (1–95). Higher = better image, higher bandwidth.",
    )

    # ------------------------------------------------------------------ #
    # Audio                                                                #
    # ------------------------------------------------------------------ #
    audio_device_index: int | None = Field(
        default=None,
        description="PyAudio device index for the USB microphone. None = system default.",
    )
    audio_sample_rate: int = Field(default=16000, description="Audio sample rate in Hz.")
    audio_channels: int = Field(default=1, ge=1, le=2, description="Mono (1) or stereo (2).")
    audio_chunk_frames: int = Field(
        default=1024,
        description="Frames per PyAudio read chunk (latency vs. CPU trade-off).",
    )

    # ------------------------------------------------------------------ #
    # ML socket (Phase 2)                                                  #
    # ------------------------------------------------------------------ #
    ml_socket_path: str = Field(
        default="/tmp/baby_monitor_ml.sock",
        description=(
            "Absolute path to the Unix domain socket written by the ML pipeline. "
            "Must match the path configured in src/ml/."
        ),
    )

    # ------------------------------------------------------------------ #
    # Alert system                                                         #
    # ------------------------------------------------------------------ #
    alert_buffer_size: int = Field(
        default=50,
        ge=1,
        description="Number of recent alert events kept in memory for backfill.",
    )
    alert_max_age_seconds: int = Field(
        default=30,
        ge=1,
        description="Events older than this are not broadcast to newly connected clients.",
    )

    # ------------------------------------------------------------------ #
    # Rate limiting                                                        #
    # ------------------------------------------------------------------ #
    rate_limit_unauthenticated: int = Field(
        default=20,
        ge=1,
        description="Max requests per minute on unauthenticated endpoints.",
    )

    @field_validator("jwt_secret")
    @classmethod
    def warn_default_secret(cls, v: str) -> str:
        if v == "CHANGE_ME_IN_PRODUCTION":
            import warnings

            warnings.warn(
                "JWT_SECRET is set to the default placeholder. "
                "Set the JWT_SECRET environment variable before deploying.",
                stacklevel=2,
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return (and cache) the singleton Settings instance.

    Call get_settings.cache_clear() in tests to force re-evaluation with
    different environment variables.
    """
    return Settings()
