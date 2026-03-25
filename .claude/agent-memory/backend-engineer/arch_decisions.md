---
name: Architecture decisions
description: Streaming format, camera backend selection, auth algorithm, alert delivery model, stream token auth
type: project
---

## Video streaming: MJPEG multipart/x-mixed-replace (Phase 1)

Chosen over WebRTC for Phase 1 simplicity. Renders in `<img>` tag natively. Latency ~200-400ms on LAN — within 1s budget. Upgrade path to WebRTC (aiortc) exists via `get_video_stream_generator()` abstraction in stream.py.

**Why:** Zero JS client code, no signalling overhead, Pi CPU headroom preserved.

## Camera backend: Picamera2 > OpenCV fallback

`stream.py` auto-detects at import time:
- Picamera2 available → use libcamera/CSI (Pi production)
- OpenCV available → use V4L2 (dev laptop)
- Neither → 503

**Why:** Allows frontend and backend development without Pi hardware attached.

## Codec: JPEG baseline

No B-frame latency. Broad hardware decode. Quality via `JPEG_QUALITY` env var (default 75).

## Audio: raw PCM audio/L16, 16kHz mono

Zero encode overhead. Decodable with Web Audio API. Content-Type carries rate/channel metadata per RFC 2586. FFmpeg transcode to Ogg/Opus available if needed.

## Auth: HS256 JWT

HTTP Basic credentials (env vars) → Bearer JWT. WebSocket: token in ?token= query param (browser limitation). Upgrade to RS256 via JWT_ALGORITHM=RS256 env var — Literal already allows it.

**Constant-time credential comparison** via hmac.compare_digest — required for timing-attack resistance.

## Stream token auth (ADR-001, TASK-105 — implemented 2026-03-25)

Browsers cannot attach Authorization headers to `<img src="...">` requests.
Chosen approach: short-lived HMAC-SHA256 signed token (not a JWT) issued by `POST /v1/auth/stream-token`.
- Format: `base64url(json_payload).base64url(hmac_sha256_sig)` — dot separator
- Payload fields: `sub`, `iat`, `exp` (Unix timestamps)
- Default TTL: 60 s (STREAM_TOKEN_TTL_SECONDS env var)
- HMAC key: STREAM_TOKEN_SECRET (falls back to JWT_SECRET if not set — independent rotation supported)
- Validation: constant-time `hmac.compare_digest` — timing-attack resistant
- Token validated **once on connection** for streaming endpoints, not per frame
- Bearer JWT kept as valid alternative on stream endpoints (no breaking change)
- `require_stream_auth` dependency in auth.py handles both credential types
- Frontend must refresh stream token every ~50 s before it expires

**Why not JWT for stream tokens:** Keeps stream token distinct and independently rotatable from the Bearer JWT. No jwt library needed for validation.

**Why not cookies (Option B in ADR-001):** SameSite=Strict breaks cross-origin LAN dev setup (port 3000 vs 8000). Secure flag unreliable without HTTPS on Pi.

## Alert delivery: asyncio, at-most-once, 30s staleness window

AlertManager singleton: set of WebSocket objects + deque(maxlen=50) ring buffer. ML socket reader is a background asyncio Task that retries every 2s on disconnect. Failed WS sends remove dead clients silently. Events > 30s old not backfilled to new clients.
