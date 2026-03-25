"""
main.py — FastAPI application entrypoint for the Baby Monitor backend.

Endpoints
---------
  POST /v1/auth/token           Issue a JWT (HTTP Basic → Bearer token)
  POST /v1/auth/stream-token    Issue a short-lived stream token (ADR-001)
  GET  /v1/stream/video         MJPEG multipart live video stream
  GET  /v1/stream/audio         Raw PCM chunked live audio stream
  GET  /v1/stream/status        Stream health: camera + mic availability
  GET  /v1/alerts/recent        Last N buffered alert events (REST)
  WS   /ws/alerts               WebSocket alert broadcast (with backfill)
  GET  /healthz                 Kubernetes/systemd liveness probe (no auth)

All endpoints except /healthz and /v1/auth/token require a valid JWT.
WebSocket endpoints receive the token as ?token=<jwt> in the URL.
Stream endpoints (/v1/stream/video, /v1/stream/audio) additionally accept
a short-lived ?stream_token=<signed> query parameter (see ADR-001).

Running locally (development):
    uvicorn src.backend.main:app --reload --port 8000

Running on Pi (production — with TLS):
    uvicorn src.backend.main:app \
        --host 0.0.0.0 --port 443 \
        --ssl-certfile /etc/ssl/baby-monitor/cert.pem \
        --ssl-keyfile  /etc/ssl/baby-monitor/key.pem
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBasicCredentials
from fastapi.websockets import WebSocket, WebSocketDisconnect

from .auth import (
    create_access_token,
    create_stream_token,
    require_jwt,
    require_jwt_ws,
    require_stream_auth,
    security,
    verify_basic_credentials,
)
from .config import get_settings
from .stream import (
    AUDIO_CONTENT_TYPE,
    MJPEG_CONTENT_TYPE,
    StreamUnavailableError,
    get_video_stream_generator,
    generate_audio_pcm,
)
from .ws_alerts import alert_manager

# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Application lifespan                                                         #
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: wire up the ML socket reader background task.
    Shutdown: cancel it cleanly.
    """
    settings = get_settings()
    logger.info(
        "Baby Monitor backend starting on %s:%d", settings.host, settings.port
    )

    # Configure the alert manager with settings values
    alert_manager._buffer_size = settings.alert_buffer_size
    alert_manager._max_age_seconds = settings.alert_max_age_seconds
    alert_manager._event_buffer.maxlen = settings.alert_buffer_size  # type: ignore[assignment]

    # Start reading from the ML Unix socket (Phase 2).
    # This is non-fatal: the socket may not exist yet if the ML process is
    # not running.  The reader task will keep retrying in the background.
    alert_manager.start_ml_reader(settings.ml_socket_path)
    logger.info("ML socket reader armed (path=%s)", settings.ml_socket_path)

    yield

    # Shutdown
    alert_manager.stop_ml_reader()
    logger.info("Baby Monitor backend stopped")


# --------------------------------------------------------------------------- #
# App instance                                                                 #
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Baby Monitor API",
    version="1.0.0",
    description=(
        "Streaming and alert API for the Raspberry Pi baby monitor. "
        "All endpoints (except /healthz and /v1/auth/token) require a Bearer JWT."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS — only allow the frontend origin (same-LAN deployment)
# In production set CORS_ORIGINS env var to the actual Pi IP/hostname.
import os

_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# --------------------------------------------------------------------------- #
# Liveness probe (no auth)                                                     #
# --------------------------------------------------------------------------- #


@app.get(
    "/healthz",
    tags=["health"],
    summary="Liveness probe",
    response_description="Always 200 if the process is alive",
)
async def healthz() -> dict[str, str]:
    """
    Lightweight liveness probe for systemd / Docker health checks.
    Does NOT check camera or mic availability — use /v1/stream/status for that.
    """
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# Authentication                                                               #
# --------------------------------------------------------------------------- #


@app.post(
    "/v1/auth/token",
    tags=["auth"],
    summary="Issue JWT",
    response_description="Bearer access token",
)
async def issue_token(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> dict[str, str]:
    """
    Exchange HTTP Basic credentials for a Bearer JWT.

    The returned token must be sent in the `Authorization: Bearer <token>`
    header on all subsequent requests, or as `?token=<token>` for WebSocket
    connections.

    Credentials are configured via MONITOR_USERNAME / MONITOR_PASSWORD
    environment variables.
    """
    username = verify_basic_credentials(credentials)
    token = create_access_token(subject=username)
    settings = get_settings()
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": str(settings.jwt_expire_minutes * 60),
    }


@app.post(
    "/v1/auth/stream-token",
    tags=["auth"],
    summary="Issue short-lived stream token",
    response_description="Signed stream token for query-param auth on stream endpoints",
    responses={
        200: {
            "description": "Stream token issued",
            "content": {
                "application/json": {
                    "example": {
                        "stream_token": "eyJzdWIiOiJhZG1pbiIsImlhdCI6MTc0...",
                        "expires_in": 60,
                    }
                }
            },
        },
        401: {"description": "Missing or invalid Bearer JWT"},
    },
)
async def issue_stream_token(
    claims: Annotated[dict, Depends(require_jwt)],
) -> dict[str, Any]:
    """
    Issue a short-lived HMAC-SHA256 stream token for use as a query parameter
    on the video and audio stream endpoints.

    **Why this endpoint exists (ADR-001)**:
    Browsers cannot attach `Authorization: Bearer` headers to `<img src="...">` or
    `<audio src="...">` requests.  Call this endpoint first (with your existing
    Bearer JWT), receive a 60-second token, and append it as:

        GET /v1/stream/video?stream_token=<token>

    **Token TTL**: 60 seconds (configured via `STREAM_TOKEN_TTL_SECONDS`).
    Refresh it before expiry — the frontend should request a new token every
    50 seconds to maintain a continuous stream without re-authentication gaps.

    **Token scope**: the token is tied to the authenticated user (`sub` claim).
    It is not single-use; a new connection opened within the TTL window using
    the same token is accepted.  Treat it like a short-lived password — do not
    log or persist it.

    Requires: `Authorization: Bearer <jwt>` header.
    """
    settings = get_settings()
    subject = claims.get("sub", "")
    token = create_stream_token(subject=subject)
    return {
        "stream_token": token,
        "expires_in": settings.stream_token_ttl_seconds,
    }


# --------------------------------------------------------------------------- #
# Video stream                                                                 #
# --------------------------------------------------------------------------- #


@app.get(
    "/v1/stream/video",
    tags=["streaming"],
    summary="MJPEG live video stream",
    response_description="multipart/x-mixed-replace MJPEG stream",
    responses={
        200: {"content": {"multipart/x-mixed-replace": {}}},
        401: {"description": "Missing or invalid token (Bearer JWT or stream_token)"},
        503: {"description": "Camera unavailable"},
    },
)
async def stream_video(
    claims: Annotated[dict, Depends(require_stream_auth)],
) -> StreamingResponse:
    """
    Stream live MJPEG video from the Pi camera.

    The response is a multipart/x-mixed-replace stream.  Set this URL as the
    `src` of an `<img>` element in the browser — no JavaScript required.

    **Authentication** (either is accepted):
    - `Authorization: Bearer <jwt>` header (existing curl/test usage)
    - `?stream_token=<signed>` query parameter (browser `<img src>` usage — see ADR-001)

    Obtain a stream token from `POST /v1/auth/stream-token`.
    Token validation is performed once on connection, not per frame.

    Target: >= 15 fps at 640x480, <= 1 second end-to-end latency on LAN.
    Returns 503 if the camera cannot be opened.
    """
    settings = get_settings()
    try:
        generator = get_video_stream_generator(
            camera_index=settings.camera_index,
            width=settings.camera_width,
            height=settings.camera_height,
            fps=settings.camera_fps,
            jpeg_quality=settings.jpeg_quality,
        )
    except StreamUnavailableError as exc:
        logger.error("Video stream unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return StreamingResponse(
        generator,
        media_type=MJPEG_CONTENT_TYPE,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx proxy buffering
        },
    )


# --------------------------------------------------------------------------- #
# Audio stream                                                                 #
# --------------------------------------------------------------------------- #


@app.get(
    "/v1/stream/audio",
    tags=["streaming"],
    summary="Raw PCM live audio stream",
    response_description="Chunked audio/L16 PCM stream",
    responses={
        200: {"content": {"audio/L16": {}}},
        401: {"description": "Missing or invalid token (Bearer JWT or stream_token)"},
        503: {"description": "Microphone unavailable"},
    },
)
async def stream_audio(
    claims: Annotated[dict, Depends(require_stream_auth)],
) -> StreamingResponse:
    """
    Stream live audio from the USB microphone as raw 16-bit signed PCM.

    The response is a chunked Transfer-Encoding stream with Content-Type
    audio/L16 (RFC 2586).  Use the Web Audio API on the client to decode.

    **Authentication** (either is accepted):
    - `Authorization: Bearer <jwt>` header (existing curl/test usage)
    - `?stream_token=<signed>` query parameter (browser usage — see ADR-001)

    Obtain a stream token from `POST /v1/auth/stream-token`.
    Token validation is performed once on connection, not per frame.

    Returns 503 if the microphone cannot be opened.
    """
    settings = get_settings()
    try:
        generator = generate_audio_pcm(
            device_index=settings.audio_device_index,
            sample_rate=settings.audio_sample_rate,
            channels=settings.audio_channels,
            chunk_frames=settings.audio_chunk_frames,
        )
    except StreamUnavailableError as exc:
        logger.error("Audio stream unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    content_type = AUDIO_CONTENT_TYPE.format(
        sample_rate=settings.audio_sample_rate,
        channels=settings.audio_channels,
    )

    return StreamingResponse(
        generator,
        media_type=content_type,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------------- #
# Stream status                                                                #
# --------------------------------------------------------------------------- #


@app.get(
    "/v1/stream/status",
    tags=["streaming"],
    summary="Stream health status",
    response_description="JSON object with camera and microphone availability",
)
async def stream_status(
    claims: Annotated[dict, Depends(require_jwt)],
) -> dict[str, Any]:
    """
    Returns current availability of the camera and microphone.

    The frontend uses this to drive the stream health indicator badge.
    Poll at a low frequency (e.g., every 5 seconds) — do not hammer.

    Response schema:
    ```json
    {
      "camera":  { "available": true,  "backend": "picamera2" },
      "audio":   { "available": false, "backend": null, "detail": "pyaudio not installed" },
      "ws_clients": 3
    }
    ```
    """
    from .stream import _PICAMERA2_AVAILABLE, _CV2_AVAILABLE, _PYAUDIO_AVAILABLE
    import importlib

    settings = get_settings()

    # Camera check
    if _PICAMERA2_AVAILABLE:
        camera_backend = "picamera2"
        camera_available = True
    elif _CV2_AVAILABLE:
        camera_backend = "opencv"
        camera_available = True
    else:
        camera_backend = None
        camera_available = False

    # Audio check
    if _PYAUDIO_AVAILABLE:
        audio_backend = "pyaudio"
        audio_available = True
        audio_detail = None
    else:
        audio_backend = None
        audio_available = False
        audio_detail = "pyaudio not installed"

    return {
        "camera": {
            "available": camera_available,
            "backend": camera_backend,
            "resolution": f"{settings.camera_width}x{settings.camera_height}",
            "fps": settings.camera_fps,
        },
        "audio": {
            "available": audio_available,
            "backend": audio_backend,
            "sample_rate": settings.audio_sample_rate,
            "channels": settings.audio_channels,
            **({"detail": audio_detail} if audio_detail else {}),
        },
        "ws_clients": alert_manager.connection_count,
    }


# --------------------------------------------------------------------------- #
# Alert REST endpoint (backfill for non-WS consumers)                         #
# --------------------------------------------------------------------------- #


@app.get(
    "/v1/alerts/recent",
    tags=["alerts"],
    summary="Recent alert events",
    response_description="List of the most recent buffered alert events",
)
async def get_recent_alerts(
    claims: Annotated[dict, Depends(require_jwt)],
    limit: int = 50,
) -> dict[str, Any]:
    """
    Return up to `limit` (max 50) most recent alert events from the in-memory
    buffer.  Useful for page-load hydration without opening a WebSocket.

    Query params:
      limit (int, default 50, max 50) — number of events to return
    """
    limit = min(limit, 50)
    return {
        "events": alert_manager.recent_events(limit=limit),
        "total_buffered": alert_manager.buffered_event_count,
    }


# --------------------------------------------------------------------------- #
# WebSocket alert stream                                                       #
# --------------------------------------------------------------------------- #


@app.websocket("/ws/alerts")
async def ws_alerts(
    websocket: WebSocket,
    claims: Annotated[dict, Depends(require_jwt_ws)],
) -> None:
    """
    WebSocket endpoint for real-time alert events.

    Connection:
        wss://<pi-host>/ws/alerts?token=<jwt>

    On connect:
        The server immediately sends all non-stale buffered events (up to
        the last 50, filtered to events <= 30 seconds old) as individual
        JSON text frames.

    Live events:
        Each new alert from the ML pipeline is forwarded within 200ms as a
        JSON text frame:
        ```json
        { "type": "cry"|"motion", "confidence": 0.87, "timestamp": "..." }
        ```

    The connection is held open.  If the server or client closes it,
    the client should reconnect with exponential back-off.
    """
    await alert_manager.connect(websocket)
    try:
        # Keep the connection alive; we only send — the client rarely sends
        # anything but we must await receive to detect disconnects.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WS alerts: unexpected close for %s: %s", websocket.client, exc)
    finally:
        alert_manager.disconnect(websocket)


# --------------------------------------------------------------------------- #
# Global error handlers                                                        #
# --------------------------------------------------------------------------- #


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
