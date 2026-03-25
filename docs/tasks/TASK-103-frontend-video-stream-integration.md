# TASK-103 — Frontend: Video Feed Display with Stream Token Auth and Auto-Reconnect
**Date**: 2026-03-25
**Owner**: frontend-engineer
**Spec**: docs/specs/baby_monitor_spec.md — Phase 1 Streaming + Video Stream Authentication
**Decision**: docs/specs/decisions/ADR-001-video-stream-auth.md
**Status**: Backlog
**Complexity**: Medium

---

## Description

Render the live MJPEG video feed in the dashboard and keep it alive through connection
drops. The stream endpoint is JWT-protected via a short-lived signed URL token (see
ADR-001); the frontend must fetch this token before mounting the `<img>` element and
refresh it before it expires.

TASK-105 (backend-engineer) must be deployed and reachable before end-to-end
integration can be tested. Frontend-engineer may build and test the component with a
mock/stub token value during TASK-105 development, then switch to live integration once
TASK-105 is done.

---

## Stream Token Flow (what to implement)

1. On component mount (or after successful login), call:
   ```
   POST /v1/auth/stream-token
   Authorization: Bearer <existing-jwt>
   ```
   Response: `{ "stream_token": "<signed>", "expires_in": 60 }`

2. Build the stream URL: `/v1/stream/video?stream_token=<token>`

3. Set that URL as the `src` on an `<img>` element. The browser opens the MJPEG
   multipart connection and renders frames as they arrive.

4. Start a refresh timer at 50 seconds (10 seconds before the 60-second TTL expires).
   On timer fire:
   - Fetch a new stream token.
   - Update `<img src>` with the new URL. This will cause the browser to close the old
     connection and open a new one — that is the intended reconnect mechanism.

5. On any `<img>` `error` event (stream dropped, server restarted, network blip):
   - Wait 2 seconds.
   - Fetch a fresh stream token.
   - Retry setting `<img src>`. Cap retries at 5; after 5 failures surface the
     "disconnected" state to the stream health indicator (TASK-104).

---

## Acceptance Criteria

- [ ] Live video feed renders in a browser tab on the same LAN within 3 seconds of
      page load under normal network conditions.
- [ ] Stream token is fetched before the `<img>` src is set; the `<img>` is never
      set to the bare stream URL without a token.
- [ ] Token refresh fires at 50 seconds; feed continues uninterrupted across the
      refresh (user sees no freeze or blank frame gap longer than 1 second).
- [ ] On connection drop, the component retries automatically within 5 seconds without
      user interaction.
- [ ] After 5 failed retry attempts, the stream health indicator state is set to
      "disconnected" (TASK-104 integration point).
- [ ] No JWT or stream token is stored in `localStorage` or `sessionStorage` — hold in
      React state / memory only.
- [ ] Component cleans up the refresh timer on unmount to avoid memory leaks.

---

## Dependencies

- Blocked by: TASK-105 must be live on dev Pi for end-to-end integration testing.
  (Can develop and mock-test in parallel with TASK-105.)
- Blocks: TASK-104 (stream health indicator needs the disconnect signal from this
  component).

---

## Definition of Done

- [ ] All acceptance criteria above pass on a real browser against the Pi.
- [ ] Works on Chrome (latest), Safari iOS (latest) — MJPEG rendering must be verified
      on iOS Safari which has historically been inconsistent.
- [ ] Token refresh cycle verified: observe no auth error in the Network tab across a
      60-second period.
- [ ] Auto-reconnect verified: kill and restart the backend stream; feed recovers
      within 5 seconds without user action.
- [ ] Code lives in `src/frontend/` only.

---

## Blockers

- TASK-105 is the hard dependency for end-to-end testing. Coordinate with
  backend-engineer; get a signal when the endpoint is deployed to dev Pi.
- iOS Safari MJPEG: if Safari does not render multipart MJPEG reliably, escalate to
  PM immediately — this may require a WebRTC fallback or a canvas-polling workaround,
  which is a scope change.

---

## Implementation Notes (not prescriptive — frontend-engineer owns the approach)

- The token refresh can be implemented with `useEffect` + `setTimeout` or a custom hook.
- Consider a `useStreamToken` hook that encapsulates fetch + refresh + cleanup, keeping
  the video component clean.
- The `<img>` reconnect-on-src-change behavior is browser-native for MJPEG; no special
  stream teardown code is needed — just update the `src`.
