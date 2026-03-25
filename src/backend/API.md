# Baby Monitor Backend — API Contract

**Version**: 1.0
**Date**: 2026-03-25
**Owner**: backend-engineer
**Audience**: frontend-engineer

This document is the authoritative handoff contract for all backend HTTP and
WebSocket endpoints.  The machine-readable version lives at
`src/backend/docs/openapi.yaml`.

---

## Base URL

| Environment | URL |
|---|---|
| Pi (HTTP, development) | `http://raspberrypi.local:8000` |
| Pi (HTTPS, production) | `https://raspberrypi.local` |

All versioned endpoints are prefixed with `/v1/`.

---

## Authentication Overview

The API uses two token types.

### Bearer JWT

- Obtained from `POST /v1/auth/token` using HTTP Basic credentials.
- Valid for 12 hours (configurable via `JWT_EXPIRE_MINUTES`).
- Send as `Authorization: Bearer <token>` on all non-streaming requests.
- Also accepted on stream endpoints for curl/CLI usage.

### Stream Token (ADR-001)

- Obtained from `POST /v1/auth/stream-token` using a valid Bearer JWT.
- Valid for **60 seconds** (configurable via `STREAM_TOKEN_TTL_SECONDS`).
- Appended as `?stream_token=<token>` on stream URLs.
- **Required for browser `<img src="...">` and `<audio src="...">` usage**
  because browsers cannot send custom headers on those requests.
- Refresh every ~50 seconds to prevent the stream from going dark mid-session.

**Why two token types?** See `docs/specs/decisions/ADR-001-video-stream-auth.md`.

---

## Endpoints

### POST /v1/auth/token

Issue a Bearer JWT using HTTP Basic credentials.

**Auth**: HTTP Basic (`MONITOR_USERNAME` / `MONITOR_PASSWORD` env vars on the Pi)
**No Bearer token required.**

#### Request

```
POST /v1/auth/token
Authorization: Basic <base64(username:password)>
```

No request body.

#### Response 200

```json
{
  "access_token": "<jwt>",
  "token_type": "bearer",
  "expires_in": "43200"
}
```

| Field | Type | Description |
|---|---|---|
| `access_token` | string | Signed JWT — pass as `Bearer <token>` |
| `token_type` | string | Always `"bearer"` |
| `expires_in` | string | Validity in seconds (string for broad client compat) |

#### Response 401

Bad credentials.

```json
{ "detail": "Incorrect username or password" }
```

---

### POST /v1/auth/stream-token

Issue a short-lived stream token for use as a `?stream_token=` query parameter
on the video and audio stream endpoints.

**Auth**: `Authorization: Bearer <jwt>` (required)

#### Request

```
POST /v1/auth/stream-token
Authorization: Bearer <jwt>
```

No request body.

#### Response 200

```json
{
  "stream_token": "<signed-token>",
  "expires_in": 60
}
```

| Field | Type | Description |
|---|---|---|
| `stream_token` | string | HMAC-SHA256 signed token — append as `?stream_token=<value>` |
| `expires_in` | integer | Validity in seconds (always 60 in current config) |

#### Response 401

Bearer JWT is missing, malformed, or expired.

```json
{ "detail": "Invalid or expired token" }
```

#### Frontend usage pattern

```js
// 1. On page load / stream mount
const { stream_token } = await fetch('/v1/auth/stream-token', {
  method: 'POST',
  headers: { Authorization: `Bearer ${jwt}` },
}).then(r => r.json());

imgEl.src = `/v1/stream/video?stream_token=${stream_token}`;

// 2. Refresh before expiry (every 50 s is safe)
setInterval(async () => {
  const { stream_token: newToken } = await fetch('/v1/auth/stream-token', {
    method: 'POST',
    headers: { Authorization: `Bearer ${jwt}` },
  }).then(r => r.json());
  imgEl.src = `/v1/stream/video?stream_token=${newToken}`;
}, 50_000);
```

---

### GET /v1/stream/video

MJPEG live video stream from the Pi camera.

**Auth**: either of:
- `Authorization: Bearer <jwt>` header
- `?stream_token=<signed>` query parameter  ← recommended for `<img src="...">`

Token is validated **once on connection**, not per frame.

#### Request

```
GET /v1/stream/video?stream_token=<signed>
```

Or with a Bearer header for non-browser clients:

```
GET /v1/stream/video
Authorization: Bearer <jwt>
```

#### Response 200

`Content-Type: multipart/x-mixed-replace; boundary=frame`

Continuous MJPEG multipart stream.  Render directly as an `<img>` src:

```html
<img src="/v1/stream/video?stream_token=TOKEN" />
```

Performance target: >= 15 fps at 640x480, <= 1 s end-to-end latency on LAN.

#### Response 401

Token absent, invalid, or expired.

```json
{ "detail": "Stream token expired" }
```

#### Response 503

Camera hardware unavailable.

```json
{ "detail": "OpenCV could not open camera index 0" }
```

---

### GET /v1/stream/audio

Raw PCM audio stream from the USB microphone.

**Auth**: either of:
- `Authorization: Bearer <jwt>` header
- `?stream_token=<signed>` query parameter

Token is validated **once on connection**.

#### Request

```
GET /v1/stream/audio?stream_token=<signed>
```

#### Response 200

`Content-Type: audio/L16;rate=16000;channels=1`

Chunked Transfer-Encoding stream of raw signed 16-bit little-endian PCM.
Sample rate and channel count come from `AUDIO_SAMPLE_RATE` / `AUDIO_CHANNELS`
env vars and are reflected in the `Content-Type` header.

Decode with the Web Audio API:

```js
// Minimal decode sketch — production code needs a proper PCM→AudioBuffer pipeline
const ctx = new AudioContext({ sampleRate: 16000 });
const response = await fetch(`/v1/stream/audio?stream_token=${token}`);
const reader = response.body.getReader();
// ... read chunks, convert Int16Array → Float32Array, schedule AudioBufferSourceNode
```

#### Response 401

Token absent, invalid, or expired.

#### Response 503

Microphone unavailable.

---

### GET /v1/stream/status

Stream health: camera and microphone availability, connected WebSocket clients.

**Auth**: `Authorization: Bearer <jwt>` (required)

#### Request

```
GET /v1/stream/status
Authorization: Bearer <jwt>
```

#### Response 200

```json
{
  "camera": {
    "available": true,
    "backend": "picamera2",
    "resolution": "640x480",
    "fps": 15
  },
  "audio": {
    "available": false,
    "backend": null,
    "sample_rate": 16000,
    "channels": 1,
    "detail": "pyaudio not installed"
  },
  "ws_clients": 2
}
```

| Field | Type | Values |
|---|---|---|
| `camera.available` | boolean | `true` if camera backend is importable |
| `camera.backend` | string\|null | `"picamera2"`, `"opencv"`, or `null` |
| `camera.resolution` | string | `"WIDTHxHEIGHT"` |
| `camera.fps` | integer | Configured target fps |
| `audio.available` | boolean | `true` if pyaudio is installed |
| `audio.backend` | string\|null | `"pyaudio"` or `null` |
| `audio.detail` | string | Human-readable reason when `available=false` |
| `ws_clients` | integer | Currently connected WebSocket alert clients |

Recommended poll interval: every 5 seconds.

#### Response 401

```json
{ "detail": "Invalid or expired token" }
```

---

### WS /ws/alerts

WebSocket endpoint for real-time ML alert events.

**Auth**: `?token=<jwt>` query parameter (browser WebSocket API does not
support custom headers).

#### Connection URL

```
wss://<pi-host>/ws/alerts?token=<jwt>
```

#### On connect — backfill

The server immediately sends all non-stale buffered events as individual JSON
text frames (up to 50 events, filtered to those within the last 30 seconds).

#### Live event frame

```json
{
  "type": "cry",
  "confidence": 0.87,
  "timestamp": "2026-03-25T02:14:33.123Z"
}
```

| Field | Type | Values |
|---|---|---|
| `type` | string | `"cry"` or `"motion"` |
| `confidence` | float | Model confidence 0.0–1.0 |
| `timestamp` | string | ISO 8601 UTC |

#### Delivery guarantees

- **At-most-once**: the server does not retry failed sends.
- **Staleness filter**: events older than 30 s are not delivered on connect.
- **Reconnect**: implement exponential back-off on the client side.

#### Response 401

WebSocket upgrade refused if token is missing or invalid.

---

### GET /v1/alerts/recent

REST fallback for page-load hydration: up to 50 buffered alert events.

**Auth**: `Authorization: Bearer <jwt>` (required)

#### Request

```
GET /v1/alerts/recent?limit=20
Authorization: Bearer <jwt>
```

Query params:

| Param | Type | Default | Max | Description |
|---|---|---|---|---|
| `limit` | integer | 50 | 50 | Number of events to return |

#### Response 200

```json
{
  "events": [
    {
      "type": "cry",
      "confidence": 0.87,
      "timestamp": "2026-03-25T02:14:33.123Z"
    }
  ],
  "total_buffered": 12
}
```

Events are ordered from oldest to newest.

#### Response 401

```json
{ "detail": "Invalid or expired token" }
```

---

### GET /healthz

Liveness probe — **no authentication required**.

Returns 200 if the process is alive.  Does NOT verify camera or mic hardware.
Use `/v1/stream/status` for hardware health.

#### Response 200

```json
{ "status": "ok" }
```

---

## Error Response Shape

All error responses use a consistent shape:

```json
{ "detail": "<human-readable message>" }
```

| HTTP status | Meaning |
|---|---|
| 401 | Token absent, malformed, or expired |
| 403 | Token valid but insufficient permissions (not currently used) |
| 503 | Hardware (camera/mic) unavailable |
| 500 | Unhandled server error |

---

## Environment Variables (relevant to the frontend)

These control the values the frontend receives or should be aware of:

| Variable | Default | Description |
|---|---|---|
| `JWT_EXPIRE_MINUTES` | `720` | Bearer JWT TTL (12 h) |
| `STREAM_TOKEN_TTL_SECONDS` | `60` | Stream token TTL — refresh before this |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `640` / `480` | Video resolution |
| `CAMERA_FPS` | `15` | Target frame rate |
| `AUDIO_SAMPLE_RATE` | `16000` | PCM sample rate in Hz |
| `AUDIO_CHANNELS` | `1` | 1 = mono, 2 = stereo |

---

## Change Log

| Date | Change | Task |
|---|---|---|
| 2026-03-25 | Add `POST /v1/auth/stream-token`; add `?stream_token` param to video and audio stream endpoints; Bearer JWT kept as valid alternative | TASK-105 |
