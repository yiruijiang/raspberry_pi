# Baby Monitor — Backend Design Document

**Date**: 2026-03-25
**Author**: backend-engineer
**Status**: Phase 1 implemented; Phase 2 wired but awaiting ML socket contract confirmation

---

## 1. Overview

The backend is a **Python 3.11 FastAPI** application that runs as a systemd
service on a Raspberry Pi 4B.  It has three responsibilities:

1. **Video streaming** — capture frames from the Pi CSI camera and deliver
   them as an MJPEG multipart HTTP stream.
2. **Audio streaming** — capture audio from a USB microphone and deliver it
   as chunked raw PCM.
3. **Alert broadcasting** — read JSON events from the ML Unix domain socket
   and fan them out to all connected WebSocket clients.

OpenAPI spec: `src/backend/docs/openapi.yaml`

---

## 2. Directory Structure

```
src/backend/
├── __init__.py
├── main.py           # FastAPI app, route handlers, lifespan
├── config.py         # Pydantic Settings (env-var based)
├── auth.py           # JWT issuance and validation
├── stream.py         # Camera capture (Picamera2 / OpenCV fallback) + audio
├── ws_alerts.py      # AlertManager — WebSocket broadcast + ML socket reader
├── .env.example      # All configurable environment variables documented
├── requirements.txt
├── docs/
│   └── openapi.yaml  # OpenAPI 3.1 spec (authoritative API contract)
└── tests/
    ├── test_config.py
    ├── test_auth.py
    └── test_ws_alerts.py
```

---

## 3. API Endpoints (v1)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET    | /healthz | None | Liveness probe |
| POST   | /v1/auth/token | Basic | Issue JWT |
| GET    | /v1/stream/video | Bearer | MJPEG live video |
| GET    | /v1/stream/audio | Bearer | Raw PCM live audio |
| GET    | /v1/stream/status | Bearer | Camera/mic availability |
| GET    | /v1/alerts/recent | Bearer | Last ≤50 buffered alerts (REST) |
| WS     | /ws/alerts | Bearer (query param ?token=) | Real-time alert stream |

Full request/response schemas: `src/backend/docs/openapi.yaml`

---

## 4. Architecture Decisions

### 4.1 Video: MJPEG over HTTP (Phase 1)

**Choice**: `multipart/x-mixed-replace` MJPEG stream via `StreamingResponse`.

**Rationale**:
- Renders natively in browser `<img>` tags — zero JavaScript.
- Delivers individual frames immediately; no segment buffer accumulation.
- On LAN: typically 200–400ms end-to-end latency, well within the 1s budget.
- Simple to implement; the Pi CPU is not burdened with WebRTC signalling.

**Codec**: JPEG baseline.
- Broadest hardware decode support.
- No B-frame latency.
- Quality/bandwidth tunable via `JPEG_QUALITY` env var (default 75).

**Future**: If latency must drop below 200ms, migrate to WebRTC
(`aiortc` library). The `get_video_stream_generator()` abstraction in
`stream.py` makes this a clean swap.

### 4.2 Camera backend: Picamera2 with OpenCV fallback

`stream.py` detects at import time which library is available:

```
Picamera2 available? → use libcamera / CSI pipeline (Pi production)
OpenCV available?    → use V4L2 capture (dev laptop / USB webcam)
Neither?             → raise StreamUnavailableError → 503
```

This lets the frontend-engineer and backend-engineer run the server locally
without a Pi attached.

### 4.3 Audio: raw PCM (audio/L16)

**Choice**: Signed 16-bit LE PCM, 16 kHz mono, chunked HTTP response.

**Rationale**:
- Universally decodable with the Web Audio API.
- Zero encode overhead on the Pi (no Ogg/Opus transcode).
- `Content-Type: audio/L16;rate=16000;channels=1` (RFC 2586) tells the
  client the exact format without negotiation.

If Ogg/Opus is needed for broader `<audio>` element compat without Web Audio
API, add an FFmpeg transcode subprocess in `stream.py`.

### 4.4 Authentication: HS256 JWT

**Choice**: JWT with HS256, issued at `/v1/auth/token` via HTTP Basic credentials.

**Why HTTP Basic for the credential exchange**: single-household deployment;
no need for OAuth flows. The Basic credentials (username + password) are set
via environment variables on the Pi.

**WebSocket tokens**: browser WebSocket API doesn't support custom headers,
so the token is passed as `?token=<jwt>` and validated before the upgrade
completes.

**Upgrade path**: switch to RS256 by setting `JWT_ALGORITHM=RS256` and
providing an RSA key pair. The `config.py` Literal already allows this.

### 4.5 Alert broadcasting: in-memory, asyncio, at-most-once

`AlertManager` (singleton in `ws_alerts.py`) maintains:
- A `set` of active `WebSocket` objects.
- A `deque(maxlen=50)` ring buffer for backfill.
- A background `asyncio.Task` that reads from the ML Unix socket.

**Delivery**: at-most-once per client. Failed sends remove the dead client
from the active set. No retry, no queue — stale events > 30s are filtered.

**Reconnect**: the ML socket reader retries with a 2-second delay on any
connection or read error. This means the backend is robust to the ML process
starting after the backend.

---

## 5. ML Socket Contract (Phase 2)

Agreed with ml-engineer. The ML pipeline writes newline-delimited JSON to a
Unix domain socket at `ML_SOCKET_PATH` (default `/tmp/baby_monitor_ml.sock`).

**Event schema**:
```json
{
  "type":       "cry" | "motion",
  "confidence": 0.0 to 1.0,
  "timestamp":  "<ISO 8601 UTC>"
}
```

Events are forwarded to WebSocket clients as-is (no transformation).

**Status**: The backend reader task is already implemented and armed at
startup. It will silently retry until the ML process creates the socket.
No backend restart is required when the ML pipeline starts.

---

## 6. Security Checklist

- [x] All endpoints except `/healthz` and `/v1/auth/token` require valid JWT
- [x] No raw tokens, passwords, or PII logged
- [x] Credentials loaded from environment variables; `.env.example` documents all of them
- [x] WebSocket token validated before upgrade accepts
- [x] Constant-time comparison for Basic credential check
- [ ] TLS: configure via `TLS_CERTFILE` / `TLS_KEYFILE` env vars (see §7)
- [ ] Rate limiting: to be added via slowapi middleware before public exposure

---

## 7. Deployment on Raspberry Pi

### Prerequisites

```bash
# Pi OS Bookworm (arm64)
sudo apt update
sudo apt install -y python3.11 python3.11-venv portaudio19-dev python3-picamera2
```

### Install Python dependencies

```bash
cd /home/pi/baby-monitor
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r src/backend/requirements.txt
```

### Environment variables

```bash
cp src/backend/.env.example src/backend/.env
# Edit .env — set JWT_SECRET, MONITOR_PASSWORD, TLS paths
```

### Self-signed TLS (LAN-only)

```bash
sudo mkdir -p /etc/ssl/baby-monitor
sudo openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout /etc/ssl/baby-monitor/key.pem \
  -out    /etc/ssl/baby-monitor/cert.pem \
  -subj "/CN=raspberrypi.local"
```

Then set in `.env`:
```
TLS_CERTFILE=/etc/ssl/baby-monitor/cert.pem
TLS_KEYFILE=/etc/ssl/baby-monitor/key.pem
```

### systemd service

See `src/backend/baby-monitor-backend.service`.

```bash
sudo cp src/backend/baby-monitor-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now baby-monitor-backend
sudo journalctl -u baby-monitor-backend -f
```

### Running manually (development)

```bash
# HTTP (dev)
uvicorn src.backend.main:app --reload --host 0.0.0.0 --port 8000

# HTTPS (production)
uvicorn src.backend.main:app \
  --host 0.0.0.0 --port 443 \
  --ssl-certfile /etc/ssl/baby-monitor/cert.pem \
  --ssl-keyfile  /etc/ssl/baby-monitor/key.pem
```

---

## 8. Running Tests

```bash
# From repo root
pip install pytest pytest-asyncio httpx
pytest src/backend/tests/ -v
```

Tests do NOT require a Pi camera or microphone — they mock hardware
dependencies and test business logic only.

---

## 9. API Change Log

| Date       | Change | Impact |
|------------|--------|--------|
| 2026-03-25 | Initial implementation: all Phase 1 endpoints + /ws/alerts skeleton | Frontend-engineer: integrate video/audio streams; see openapi.yaml for schemas |

---

## 10. Open Questions / Dependencies

1. **ML socket path**: default is `/tmp/baby_monitor_ml.sock`. Confirm with
   ml-engineer before Phase 2 integration starts.
2. **Audio format**: currently raw PCM (audio/L16). If the frontend team
   finds Web Audio API decoding difficult, we can switch to Ogg/Opus via
   FFmpeg — needs a backend change + frontend re-integration.
3. **Authentication for /v1/stream/video in `<img>` tags**: browsers do not
   send `Authorization` headers with `<img>` requests. Current workaround:
   embed the token in a short-lived signed URL, or use a session cookie.
   This needs coordination with the frontend-engineer before TASK-103.
4. **Rate limiting**: `slowapi` middleware should be added before the monitor
   is exposed beyond the immediate Pi LAN.
5. **Alert history persistence**: currently in-memory (lost on restart). If
   the PM wants history to survive restarts, add a SQLite store (Phase 3).
