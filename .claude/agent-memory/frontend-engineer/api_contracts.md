---
name: Confirmed backend API contracts
description: API endpoints confirmed in openapi.yaml and ADR-001 that frontend consumes
type: project
---

Authoritative contract source: src/backend/docs/openapi.yaml

## Confirmed endpoints (as of 2026-03-25)

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| /v1/auth/token | POST | HTTP Basic | Get Bearer JWT |
| /v1/stream/video | GET | stream_token query param | MJPEG video stream |
| /v1/stream/audio | GET | stream_token query param | PCM audio stream |
| /v1/stream/status | GET | Bearer JWT | Camera/mic health poll (5s interval) |
| /v1/alerts/recent | GET | Bearer JWT | Page-load alert hydration (max 50) |
| /ws/alerts | WS | ?token=<jwt> query param | Real-time alert events |

## /v1/auth/stream-token (confirmed deployed — TASK-105 complete as of 2026-03-25)

| Endpoint | Method | Auth | Response |
|---|---|---|---|
| /v1/auth/stream-token | POST | Bearer JWT | { stream_token: string, expires_in: 60 } |

- expires_in is an integer (not a string) — confirmed in API.md
- Refresh every 50s (10s before 60s TTL). Documented in API.md frontend usage pattern.
- VITE_MOCK_STREAM_TOKEN removed from codebase; mock block removed from useStreamToken.

## WebSocket alert event schema

```json
{ "type": "cry" | "motion", "confidence": 0.0-1.0, "timestamp": "ISO8601" }
```

On connect: server sends up to 50 buffered events from last 30 seconds.
Delivery: at-most-once per client, no retry.

## Token response schema

POST /v1/auth/token returns:
```json
{ "access_token": "<jwt>", "token_type": "bearer", "expires_in": "<seconds as string>" }
```
Note: expires_in is a STRING (not number) — backend does this for broad client compat.

**Why:** Backend engineer published openapi.yaml; ADR-001 approved signed URL token flow for stream auth.
