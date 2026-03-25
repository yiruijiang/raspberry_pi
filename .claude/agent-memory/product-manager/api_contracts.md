---
name: API Contracts
description: Finalized inter-team communication schemas: ML→Backend socket, Backend→Frontend HTTP and WebSocket
type: project
---

These contracts were established in baby_monitor_spec.md (2026-03-25) and must not
change without coordinating all three teams simultaneously.

**Why:** Frontend and backend cannot be built in parallel with ML unless the schema is
stable. Any change to the socket schema is a multi-team breaking change.

**How to apply:** Before accepting any ML engineer change to output format, require
explicit sign-off from backend-engineer. Before accepting any backend WS format change,
require sign-off from frontend-engineer.

## ML → Backend (Unix domain socket)
JSON, one object per line, newline-delimited.
```
{
  "type": "cry" | "motion",
  "confidence": float,        // 0.0–1.0
  "timestamp": "<ISO8601>"    // e.g. "2026-03-25T02:14:33Z"
}
```
Socket path configurable via environment variable.

## Backend → Frontend (WebSocket)
Endpoint: `WS /ws/alerts`
Forwards ML events as-is (same JSON schema).
On new client connect: server sends last 50 buffered events.

## Backend HTTP Streaming
- `GET /stream/video` — multipart MJPEG (`Content-Type: multipart/x-mixed-replace`)
- `GET /stream/audio` — chunked Ogg/Opus or PCM, browser-playable
