# TASK-105 — Backend: Stream Token Endpoint and Query-Param Auth on Stream Routes
**Date**: 2026-03-25
**Owner**: backend-engineer
**Spec**: docs/specs/baby_monitor_spec.md — "Video Stream Authentication"
**Decision**: docs/specs/decisions/ADR-001-video-stream-auth.md
**Status**: Backlog
**Complexity**: Small

---

## Description

The browser cannot send `Authorization: Bearer` headers on `<img src>` requests, so
the MJPEG video stream (and audio stream) cannot use the existing Bearer token guard
directly. This task adds:

1. A new endpoint `POST /v1/auth/stream-token` that accepts a valid Bearer JWT and
   returns a short-lived signed stream token.
2. Query-param token validation on the two stream routes.

This unblocks TASK-103 (frontend video stream integration).

---

## Acceptance-Level Contract

These are the exact contracts the frontend will code against. Do not deviate without
coordinating with the frontend-engineer and updating the spec.

### POST /v1/auth/stream-token

- **Auth**: `Authorization: Bearer <jwt>` (existing JWT guard, no changes)
- **Request body**: none
- **Response 200**:
  ```json
  { "stream_token": "<signed-string>", "expires_in": 60 }
  ```
- **Response 401**: invalid or expired JWT — existing error shape.

### GET /v1/stream/video and GET /v1/stream/audio

- Accept optional `?stream_token=<signed>` query parameter.
- If present and valid: allow stream.
- If absent or invalid/expired: return `401` before streaming begins.
- Token is validated on connection establishment only — not re-checked per frame while
  the connection is open.

### Token Properties

- TTL: 60 seconds from issuance.
- Signing: HMAC-SHA256 with a server-side secret (can share the JWT secret or use a
  dedicated one — backend's choice, document in design doc).
- Payload must encode at minimum: `{"exp": <unix timestamp>}`. Optionally a `sub` to
  scope the token to a specific stream type, but not required for v1.
- Tokens are not single-use (re-using within TTL window is fine; the stream reconnects
  and re-validates).

---

## Dependencies

- Blocked by: nothing — can start immediately.
- Blocks: TASK-103 (frontend cannot integrate stream until this endpoint exists and is
  documented with an example response).

---

## Definition of Done

- [ ] `POST /v1/auth/stream-token` returns a signed token for any valid Bearer JWT.
- [ ] `GET /v1/stream/video` rejects requests without a valid `stream_token` param with
      `401` before any stream bytes are written.
- [ ] `GET /v1/stream/audio` same behavior as video.
- [ ] Token expiry is enforced: a token issued > 60 seconds ago is rejected.
- [ ] Unit tests cover: valid token accepted, expired token rejected, no token rejected,
      tampered token rejected.
- [ ] Response schema matches the contract above exactly.
- [ ] Backend engineer documents the signing approach in `docs/design/` (their doc, not
      the PM's spec).

---

## Blockers

None known. This is greenfield — no existing stream-auth code to conflict with.

**Dispatch note**: This task is PARALLEL-safe with TASK-103 only up to the point where
frontend needs to integrate. Frontend-engineer can build the UI scaffolding and mock
the token, but cannot do end-to-end integration until this endpoint is deployed and
reachable on the Pi. Signal frontend-engineer when the endpoint is live on the dev Pi.
