---
# ADR-001 — Video Stream Authentication Method
**Date**: 2026-03-25
**Status**: Accepted
**Decider**: product-manager
**Affects**: backend-engineer, frontend-engineer
**Unblocks**: TASK-103 (frontend video stream integration)

---

## Context

The backend exposes `GET /v1/stream/video` as a protected MJPEG multipart stream
requiring JWT Bearer authentication. Browsers cannot attach `Authorization: Bearer`
headers to `<img src="...">` requests; this is a hard browser constraint. Two
mechanisms were evaluated to close the gap.

## Options Considered

### Option A — Signed URL token (query param)
Frontend calls `POST /v1/auth/stream-token` with its existing Bearer token, receives
a short-lived signed token, and appends it as `?stream_token=<signed>` to the `<img>`
src. Backend validates the query param on each stream connection.

### Option B — httpOnly session cookie
Backend sets an `httpOnly SameSite=Strict` cookie at login. Browser sends it
automatically on all requests including `<img src>`.

## Decision

**Option A — Signed URL token.**

## Rationale

1. **Same-origin is not guaranteed on LAN.** The React PWA dev server (`:3000`) and
   the FastAPI backend (`:8000`) typically run on different ports, making them
   cross-origin. `SameSite=Strict` cookies will not be sent cross-origin. Aligning
   them requires either a reverse proxy (added infrastructure) or relaxing `SameSite`
   to `Lax`/`None`, which weakens the security posture.

2. **HTTPS not available on LAN.** Modern browsers restrict `Secure` cookies and
   increasingly refuse to set cookies on non-secure (`http://`) origins. Without
   HTTPS on the Pi, cookie-based auth is fragile and browser-version-sensitive.

3. **Stateless and simple.** A signed URL token is validated purely by the backend
   without session store, cookie jar management, or CORS preflight coordination.

4. **Acceptable security surface.** This is a LAN-only deployment on a trusted home
   network. A 60-second TTL signed token that leaks gives an attacker a narrow,
   expiring window — an acceptable trade-off for the deployment context.

5. **Frontend implementation is minimal.** One additional `POST` call before setting
   `<img src>`. No cookie headers, no CORS credential flags, no service worker
   intercept needed.

## Consequences

- Backend must add `POST /v1/auth/stream-token` endpoint and query-param validation
  on the stream route. See task TASK-105.
- Frontend must call the token endpoint before mounting the video element and refresh
  the token before it expires. See task TASK-103 (updated) / TASK-106.
- The audio stream endpoint (`GET /v1/stream/audio`) must apply the same pattern if
  it is also protected. Assumed yes; backend to confirm.
- Stream token is single-use or connection-scoped (backend decision); document in
  the backend design doc.

## What Is NOT Changing

- The existing JWT Bearer flow for all non-streaming endpoints is unchanged.
- The WebSocket alert endpoint (`WS /ws/alerts`) uses the existing Bearer token via
  the `Sec-WebSocket-Protocol` header trick or query param — this ADR does not cover
  WebSocket auth; that is already handled.

---
