# Baby Monitor — Phased Task Plan
**Date**: 2026-03-25
**Status**: Approved
**Author**: product-manager
**Spec**: [baby_monitor_spec.md](baby_monitor_spec.md)

---

## Dispatch Rules

Phases are sequential. Within a phase, tasks marked **[PARALLEL]** may be dispatched
simultaneously. Tasks marked **[SEQUENTIAL]** must wait for their listed dependency.

Phase 1 must be complete before Phase 2 begins.
Phase 2 must be complete before Phase 3 begins.

---

## Phase 1 — Streaming

**Goal**: Parents can open a browser on their home network and see and hear their baby
live, with automatic reconnection.

**Entry condition**: Pi camera and USB mic are physically attached and enabled.

---

### TASK-101: Video Capture and MJPEG Stream Endpoint [PARALLEL]
**Owner**: backend-engineer
**File scope**: `src/backend/`
**Complexity**: Medium
**Spec**: baby_monitor_spec.md — Phase 1, Streaming

**Description**
Implement a FastAPI route `GET /stream/video` that captures frames from the Pi camera
using Picamera2, encodes them as MJPEG, and delivers them as a multipart HTTP stream.
The endpoint must sustain at least 15 fps at 640x480 resolution under normal Pi load.

**Dependencies**
- Blocked by: Pi camera physically connected and enabled (hardware precondition)
- Blocks: TASK-103 (frontend video display), TASK-104 (stream health indicator)

**Definition of Done**
- [ ] `GET /stream/video` returns `Content-Type: multipart/x-mixed-replace`
- [ ] Feed renders at >= 15 fps in a browser on the same LAN
- [ ] Endpoint recovers gracefully if the camera is temporarily unavailable (returns 503,
      does not crash the server)
- [ ] Latency from capture to browser render is <= 1 second on LAN

---

### TASK-102: Audio Capture and Streaming Endpoint [PARALLEL]
**Owner**: backend-engineer
**File scope**: `src/backend/`
**Complexity**: Medium
**Spec**: baby_monitor_spec.md — Phase 1, Streaming

**Description**
Implement a FastAPI route `GET /stream/audio` that captures audio from the USB
microphone and delivers it as a chunked stream (Ogg/Opus or raw PCM). The browser
`<audio>` element must be able to play the stream directly.

**Dependencies**
- Blocked by: USB microphone physically attached (hardware precondition)
- Blocks: TASK-103 (frontend audio element)

**Definition of Done**
- [ ] `GET /stream/audio` returns a browser-playable chunked audio stream
- [ ] Audio plays in a browser tab on the same LAN with <= 500ms lag behind video
- [ ] Endpoint returns 503 gracefully if mic is unavailable

---

### TASK-103: Frontend Feed Display with Auto-Reconnect [SEQUENTIAL — after TASK-101 and TASK-102]
**Owner**: frontend-engineer
**File scope**: `src/frontend/`
**Complexity**: Medium
**Spec**: baby_monitor_spec.md — Phase 1, Streaming

**Description**
Build the main feed view in React. Render the MJPEG video stream in an `<img>` tag and
the audio stream in an `<audio>` tag. Implement auto-reconnect logic: if either stream
drops, retry with exponential backoff (max 5s interval) and update the stream health
indicator accordingly.

**Dependencies**
- Blocked by: TASK-101 (video endpoint live), TASK-102 (audio endpoint live)
- Blocks: TASK-104 (health indicator UX), Phase 2 frontend tasks

**Definition of Done**
- [ ] Video feed renders within 3 seconds of page load on LAN
- [ ] Audio plays alongside video within 500ms sync tolerance
- [ ] If either stream drops, client retries automatically within 5 seconds without user
      action
- [ ] Feed view is responsive and usable on a 375px-wide portrait viewport

---

### TASK-104: Stream Health Indicator [SEQUENTIAL — after TASK-103]
**Owner**: frontend-engineer
**File scope**: `src/frontend/`
**Complexity**: Small
**Spec**: baby_monitor_spec.md — Phase 1, Streaming

**Description**
Add a visual status badge to the feed view. States: Connected (green), Reconnecting
(amber, with spinner), Disconnected (red). The badge must reflect the actual connection
state of both stream connections (video AND audio — show worst state).

**Dependencies**
- Blocked by: TASK-103 (stream + reconnect logic must exist)
- Blocks: nothing

**Definition of Done**
- [ ] Badge renders in all three states
- [ ] State transitions are driven by actual stream connection events, not timers
- [ ] Badge is visible without scrolling on both desktop and mobile viewports

---

### Phase 1 Exit Criteria
- [ ] All four tasks above are Done
- [ ] End-to-end demo: browser on a phone loads the feed over WiFi, stream plays, kill
      the backend process, stream reconnects automatically within 5 seconds when restarted

---

## Phase 2 — ML Alerts

**Goal**: Parents receive timely, low-false-positive alerts when the baby cries or makes
significant movement.

**Entry condition**: Phase 1 is complete. Pi camera and USB mic confirmed working.

---

### TASK-201: Cry Detection Model and Inference Pipeline [PARALLEL]
**Owner**: ml-engineer
**File scope**: `src/ml/`
**Complexity**: Large
**Spec**: baby_monitor_spec.md — Phase 2, ML Alerts

**Description**
Build an audio classification pipeline using TensorFlow Lite. The model must distinguish
infant cry from background noise (TV, speech, white noise). Output inference results as
JSON to a Unix domain socket at a path configurable via environment variable.
Output schema: `{ "type": "cry", "confidence": float, "timestamp": "<ISO8601>" }`

Assemble or source a labeled dataset for validation. Target >= 85% precision on the
held-out set. Document the dataset source, preprocessing steps, and model architecture
in `src/ml/cry_detection/README.md`.

**Dependencies**
- Blocked by: USB mic working (Phase 1 precondition satisfied)
- Blocks: TASK-203 (backend alert forwarding — needs socket contract finalized)

**Definition of Done**
- [ ] Inference pipeline runs continuously on Pi without exceeding 30% CPU on a single
      core (leaves headroom for other processes)
- [ ] Output JSON written to Unix socket matches the schema above
- [ ] Cry detection precision >= 85% on held-out test set
- [ ] False positive rate on 10 minutes of typical ambient household audio < 2 events
- [ ] Model loads and begins inference within 10 seconds of process start

---

### TASK-202: Motion Detection Pipeline [PARALLEL]
**Owner**: ml-engineer
**File scope**: `src/ml/`
**Complexity**: Medium
**Spec**: baby_monitor_spec.md — Phase 2, ML Alerts

**Description**
Build a motion detection pipeline using OpenCV background subtraction (MOG2 or KNN).
Trigger a motion event only when contour area exceeds a configurable pixel threshold
(default tuned to ~10cm displacement at typical crib camera distance).
Write output to the same Unix domain socket as cry detection.
Output schema: `{ "type": "motion", "confidence": float, "timestamp": "<ISO8601>" }`
Confidence for motion events can be expressed as normalized contour area (0–1).

**Dependencies**
- Blocked by: Pi camera working (Phase 1 precondition satisfied)
- Blocks: TASK-203

**Definition of Done**
- [ ] Motion events fire on intentional limb movement; do not fire on minor pixel noise
      or camera sensor noise
- [ ] Output JSON written to Unix socket matches the schema above
- [ ] Pipeline runs at >= 10 fps analysis rate on Pi without exceeding 20% CPU on a
      single core
- [ ] Displacement threshold is configurable via environment variable or config file

---

### TASK-203: Backend Alert Ingestion and WebSocket Broadcast [SEQUENTIAL — after TASK-201 and TASK-202 socket contract is finalized]
**Owner**: backend-engineer
**File scope**: `src/backend/`
**Complexity**: Medium
**Spec**: baby_monitor_spec.md — Phase 2, ML Alerts

**Description**
Implement a background task in FastAPI that reads JSON events from the ML Unix domain
socket and broadcasts them to all connected WebSocket clients via `WS /ws/alerts`.
Forward the event JSON as-is (no transformation required). Maintain an in-memory buffer
of the last 50 events for clients that connect after events have fired.

**Dependencies**
- Blocked by: TASK-201 and TASK-202 (Unix socket schema must be finalized and stable
  before backend wires up — coordinate with ml-engineer to confirm schema before starting)
- Blocks: TASK-204 (frontend alert subscription)

**Definition of Done**
- [ ] `WS /ws/alerts` endpoint accepts WebSocket connections
- [ ] Events from ML socket are forwarded to all connected clients within 200ms of
      receipt
- [ ] New clients receive the last 50 buffered events on connection
- [ ] Server handles 0 connected clients gracefully (no crash, events still buffered)
- [ ] Server handles client disconnect gracefully (no memory leak on client list)

---

### TASK-204: Frontend Alert Subscription and Alert Feed [SEQUENTIAL — after TASK-203]
**Owner**: frontend-engineer
**File scope**: `src/frontend/`
**Complexity**: Medium
**Spec**: baby_monitor_spec.md — Phase 2, ML Alerts

**Description**
Connect the frontend to `WS /ws/alerts`. Display incoming alert events in an alert feed
panel alongside the video feed. Each alert item must show: type (cry / motion), confidence
score (e.g., "87%"), and timestamp. Implement WebSocket reconnect on disconnect (same
backoff strategy as TASK-103). On connect, render the 50-event backfill.

**Dependencies**
- Blocked by: TASK-203 (WebSocket endpoint must be live)
- Blocks: TASK-205 (threshold settings UI — needs alert type definitions)

**Definition of Done**
- [ ] Alert feed renders new events in real time with <= 500ms end-to-end latency
      (ML inference complete to visible in UI)
- [ ] Confidence score displayed per alert
- [ ] Backfill of last 50 events rendered on page load
- [ ] WebSocket reconnects automatically on drop

---

### TASK-205: Alert Threshold Settings [SEQUENTIAL — after TASK-204]
**Owner**: frontend-engineer
**File scope**: `src/frontend/`
**Complexity**: Small
**Spec**: baby_monitor_spec.md — Phase 2, ML Alerts

**Description**
Add a settings panel with per-alert-type confidence threshold sliders (cry threshold,
motion threshold, each 0–100%). Alerts below the threshold are received but not surfaced
in the UI (client-side filtering). Persist threshold values in localStorage. The filter
must apply to the live stream without a page reload.

**Dependencies**
- Blocked by: TASK-204 (alert feed must exist)
- Blocks: nothing

**Definition of Done**
- [ ] Two threshold sliders: one for cry, one for motion
- [ ] Alerts below threshold are hidden in the feed in real time
- [ ] Threshold values persist across browser sessions via localStorage
- [ ] Settings panel is reachable within 2 taps from the main feed view

---

### Phase 2 Exit Criteria
- [ ] All five tasks above are Done
- [ ] End-to-end demo: play a baby cry audio sample near the Pi mic, verify alert appears
      in the browser within 500ms
- [ ] Wave hand in front of camera, verify motion alert fires; small finger twitch does not

---

## Phase 3 — UI Polish

**Goal**: The dashboard is production-quality: intuitive, mobile-first, installable,
and kind to sleep-deprived eyes.

**Entry condition**: Phase 2 is complete.

---

### TASK-301: Dashboard Layout Refinement [PARALLEL]
**Owner**: frontend-engineer
**File scope**: `src/frontend/`
**Complexity**: Medium
**Spec**: baby_monitor_spec.md — Phase 3, UI Polish

**Description**
Polish the overall dashboard layout. Desktop: two-column layout with video feed on the
left (70% width) and alert feed on the right. Mobile portrait: stacked layout with
feed on top, alert feed scrollable below. Add a persistent bottom navigation bar for
mobile with tabs: Feed, Alerts, Settings. All interactive elements must have minimum
44x44px touch targets.

**Dependencies**
- Blocked by: Phase 2 complete (all UI components must exist before layout polish)
- Blocks: TASK-303 (PWA — manifest should reference finalized layout/icons)

**Definition of Done**
- [ ] Desktop two-column layout renders correctly at >= 1024px width
- [ ] Mobile stacked layout renders correctly at 375px width (portrait)
- [ ] Bottom nav bar present on mobile, standard top nav on desktop
- [ ] All tap targets >= 44x44px
- [ ] No horizontal scroll on 375px viewport

---

### TASK-302: Night Mode [PARALLEL]
**Owner**: frontend-engineer
**File scope**: `src/frontend/`
**Complexity**: Small
**Spec**: baby_monitor_spec.md — Phase 3, UI Polish

**Description**
Implement a night mode theme: dark background (#1a1a1a or similar), warm-tinted text
(avoid pure white), reduced brightness on the video feed overlay elements. Toggle must
be reachable within 2 taps from the main feed view. Persist preference in localStorage.
Also respond to the OS `prefers-color-scheme: dark` media query as the default state.

**Dependencies**
- Blocked by: Phase 2 complete
- Blocks: nothing (cosmetic, no downstream deps)

**Definition of Done**
- [ ] Night mode toggle reachable within 2 taps from feed view
- [ ] Preference persists across browser sessions
- [ ] OS dark mode preference applied as default on first visit
- [ ] All text in night mode meets WCAG AA contrast ratio (>= 4.5:1)
- [ ] Video feed element itself is not dimmed (only UI chrome)

---

### TASK-303: PWA Packaging [SEQUENTIAL — after TASK-301]
**Owner**: frontend-engineer
**File scope**: `src/frontend/`
**Complexity**: Medium
**Spec**: baby_monitor_spec.md — Phase 3, UI Polish

**Description**
Configure the Vite PWA plugin (vite-plugin-pwa / Workbox) to generate a service worker
and web app manifest. The app must be installable to the phone home screen and display
a splash screen. The offline shell (app chrome without live data) must load when the Pi
is unreachable, showing a clear "Monitor offline" state rather than a blank screen.

**Dependencies**
- Blocked by: TASK-301 (layout must be finalized before manifest icons and screenshots
  are generated)
- Blocks: nothing

**Definition of Done**
- [ ] App is installable via browser "Add to Home Screen" on iOS Safari and Android Chrome
- [ ] Lighthouse PWA audit score >= 90
- [ ] Offline shell renders with "Monitor offline" message when Pi is unreachable
- [ ] App manifest includes correct name, icons (192px and 512px), and theme color
- [ ] Service worker does not cache video/audio stream URLs (live data must always be
      fetched from network)

---

### TASK-304: Alert History Log [PARALLEL]
**Owner**: frontend-engineer
**File scope**: `src/frontend/`
**Complexity**: Small
**Spec**: baby_monitor_spec.md — Phase 3, UI Polish

**Description**
Add a dedicated Alert History view (accessible from the bottom nav / top nav). Display
the last 50 alert events in reverse-chronological order. Each row: type icon, type label,
confidence percentage, relative timestamp (e.g., "3 min ago") with absolute time on
hover/tap. The list is populated from the 50-event backfill on WebSocket connect and
updated live as new events arrive.

**Dependencies**
- Blocked by: Phase 2 complete (alert data pipeline must be working)
- Blocks: nothing

**Definition of Done**
- [ ] History view shows up to 50 events in reverse-chronological order
- [ ] Each event shows type, confidence, and timestamp
- [ ] List updates live as new alerts arrive without full re-render flash
- [ ] View is accessible from main nav within 1 tap

---

### Phase 3 Exit Criteria
- [ ] All four tasks above are Done
- [ ] Lighthouse PWA score >= 90
- [ ] Night mode usable in a dark room: no harsh white elements
- [ ] Full end-to-end walkthrough on a physical iPhone and Android device:
      install PWA, view feed, receive alert, check history, toggle night mode

---

## Summary Table

| Task | Owner | Phase | Complexity | Parallel? |
|---|---|---|---|---|
| TASK-101: Video stream endpoint | backend-engineer | 1 | Medium | Yes |
| TASK-102: Audio stream endpoint | backend-engineer | 1 | Medium | Yes |
| TASK-103: Frontend feed + reconnect | frontend-engineer | 1 | Medium | After 101+102 |
| TASK-104: Stream health indicator | frontend-engineer | 1 | Small | After 103 |
| TASK-201: Cry detection pipeline | ml-engineer | 2 | Large | Yes |
| TASK-202: Motion detection pipeline | ml-engineer | 2 | Medium | Yes |
| TASK-203: Backend alert ingestion + WS | backend-engineer | 2 | Medium | After 201+202 schema |
| TASK-204: Frontend alert feed | frontend-engineer | 2 | Medium | After 203 |
| TASK-205: Threshold settings UI | frontend-engineer | 2 | Small | After 204 |
| TASK-301: Dashboard layout | frontend-engineer | 3 | Medium | Yes |
| TASK-302: Night mode | frontend-engineer | 3 | Small | Yes |
| TASK-303: PWA packaging | frontend-engineer | 3 | Medium | After 301 |
| TASK-304: Alert history log | frontend-engineer | 3 | Small | Yes |

---

## Critical Path

```
TASK-101 ─┐
           ├─► TASK-103 ─► TASK-104  [Phase 1 done]
TASK-102 ─┘

TASK-201 ─┐
           ├─► TASK-203 ─► TASK-204 ─► TASK-205  [Phase 2 done]
TASK-202 ─┘

TASK-301 ─► TASK-303
TASK-302 (independent)                             [Phase 3 done]
TASK-304 (independent)
```

The critical path through the entire project is:
**TASK-101 → TASK-103 → [Phase 1] → TASK-201 → TASK-203 → TASK-204 → [Phase 2] → TASK-301 → TASK-303 → [Phase 3 done]**

The longest pole is TASK-201 (cry detection, Large complexity). ML engineer should start
this immediately when Phase 2 opens.
