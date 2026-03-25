# Baby Monitor — Full Product Spec
**Date**: 2026-03-25
**Status**: Approved
**Author**: product-manager

---

## Problem Statement

Parents of infants cannot maintain constant physical presence in a baby's room. Existing
consumer monitors are either closed-ecosystem (no customization), lack intelligent alerts
(every sound triggers a ping), or are cloud-dependent (privacy risk, latency). Parents
need a self-hosted, low-latency monitor that sees and hears what matters — and stays
quiet when it doesn't. Missed cry events or undetected positional hazards during sleep
are real safety risks. Fatigue compounds the problem: exhausted parents need the system
to do the filtering, not them.

---

## Goals

1. Deliver real-time video and audio from a Raspberry Pi camera/mic to any web browser
   or mobile device on the home network.
2. Use on-device ML to detect crying and significant motion, sending push alerts only
   when confidence thresholds are met (minimize false positives).
3. Provide a clean, usable dashboard that works well in low-light conditions and is
   accessible with one hand at 3am.

---

## Non-Goals (Out of Scope for v1)

- Cloud hosting or remote access outside the home network
- Multi-camera support
- Heart rate or breathing rate biometric detection
- Integration with smart home platforms (HomeKit, Google Home, Alexa)
- Native iOS/Android apps (PWA covers mobile)
- Recording or clip storage
- Two-way audio (talk-back)

---

## User Stories

### Streaming
- As a parent, I want to see a live video feed from my baby's room on my phone, so that
  I can check in without entering the room and risking waking the baby.
- As a parent, I want audio streaming alongside video, so that I can hear ambient sounds
  without relying solely on visual cues.
- As a parent, I want the feed to load within 3 seconds of opening the app, so that I
  do not waste critical response time.
- As a parent, I want the feed to reconnect automatically if the connection drops, so
  that I do not have to manually refresh at 2am.

### ML Alerts
- As a parent, I want to receive an in-app alert when the baby is crying, so that I
  know to respond even when my phone is in another room and the volume is low.
- As a parent, I want motion alerts only for significant movement (not minor twitches),
  so that I am not woken up unnecessarily by normal sleep movements.
- As a parent, I want alert confidence scores visible in the UI, so that I can calibrate
  my trust in the system over time.
- As a parent, I want to configure per-alert sensitivity thresholds, so that I can tune
  false-positive rates to my baby's behavior patterns.

### Dashboard and Usability
- As a parent, I want a night mode with reduced brightness and warm tones, so that
  checking the monitor does not disrupt my own sleep adaptation.
- As a parent, I want to see a recent alert history log, so that I can review what
  happened during a nap without relying on memory.
- As a parent, I want the app to be installable on my phone home screen (PWA), so that
  I have one-tap access without navigating a browser.
- As a parent, I want the dashboard to be operable with one hand in portrait mode, so
  that I can check on the baby while holding them.

---

## Core Features

### Phase 1 — Streaming
| Feature | Description |
|---|---|
| MJPEG/WebRTC video stream | Live video from Pi camera to browser |
| Audio stream | Microphone audio capture and delivery |
| Auto-reconnect | Client reconnects on dropped connection without user action |
| Stream health indicator | Visual latency/status badge on the UI |

### Phase 2 — ML Alerts
| Feature | Description |
|---|---|
| Cry detection | Audio ML model classifies infant cry vs. background noise |
| Motion detection | CV model flags significant positional movement in frame |
| Confidence scoring | Each alert carries a 0–1 confidence value |
| Configurable thresholds | Per-alert-type sensitivity slider in settings |
| In-app push alerts | WebSocket-delivered alert messages rendered in UI |

### Phase 3 — UI Polish
| Feature | Description |
|---|---|
| Dashboard layout | Feed + alert feed side-by-side (desktop) / stacked (mobile) |
| Night mode | Dark theme with reduced blue light, persisted via localStorage |
| Alert history log | Scrollable log of last 50 alert events with timestamps |
| PWA packaging | Installable manifest, service worker, offline shell |
| One-handed mobile layout | Portrait-optimized tap targets, bottom nav bar |

---

## Tech Stack Decisions

### Hardware
- **Raspberry Pi 4B** (2GB+ RAM recommended) as the capture and inference node
- **Pi Camera Module v2** or compatible CSI camera
- **USB microphone** or Pi HAT with audio capture

### ML (src/ml/)
- **Python 3.11** runtime on Pi
- **TensorFlow Lite** for on-device inference (cry detection model)
- **OpenCV** for frame capture and motion detection (background subtraction)
- Inference runs in a dedicated process; results published to a local Unix socket or
  named pipe consumed by the backend

### Backend (src/backend/)
- **Python 3.11 + FastAPI** for HTTP and WebSocket API server
- **FFmpeg** or **Picamera2** for video capture and MJPEG encoding
- **WebSocket** endpoint for real-time alert push to connected clients
- Runs as a systemd service on the Pi

### Frontend (src/frontend/)
- **React 18 + TypeScript** (Vite build toolchain)
- **WebSocket client** for alert subscription
- **PWA** via Vite PWA plugin (Workbox service worker)
- Served as static files by the FastAPI backend (or Nginx on Pi)
- No external dependencies on CDN-hosted resources (must work offline on LAN)

### Communication Contracts (inter-team)
- ML → Backend: ML process writes JSON event messages to a Unix domain socket.
  Schema: `{ "type": "cry" | "motion", "confidence": float, "timestamp": ISO8601 }`
- Backend → Frontend: WebSocket messages, same schema, forwarded as-is.
- Video: Backend exposes `GET /stream/video` as MJPEG multipart stream.
- Audio: Backend exposes `GET /stream/audio` as chunked PCM/Ogg stream.
- Alert subscription: Frontend connects to `WS /ws/alerts`.

---

## Acceptance Criteria

### Phase 1 — Streaming
- [ ] Live video feed renders in a browser tab on the same LAN within 3 seconds of page
      load under normal network conditions.
- [ ] Audio stream plays in sync with video within 500ms.
- [ ] Client automatically reconnects to the video and audio streams within 5 seconds
      of a dropped connection, without user interaction.
- [ ] Stream health indicator accurately reflects connected / reconnecting / disconnected
      state.

### Phase 2 — ML Alerts
- [ ] Cry detection achieves >= 85% precision on a held-out test set of infant cry
      recordings (to be assembled by ML engineer).
- [ ] Motion detection triggers on intentional limb movement > 10cm displacement in
      frame and does not trigger on minor twitches or camera noise.
- [ ] Alert events arrive in the browser UI within 500ms of ML inference completing.
- [ ] Confidence score is displayed alongside each alert in the UI.
- [ ] Threshold sliders in settings take effect on the next inference cycle without
      requiring a server restart.

### Phase 3 — UI Polish
- [ ] Night mode toggle is reachable within 2 taps from the main feed view and persists
      across sessions.
- [ ] Alert history shows the last 50 events with type, confidence, and timestamp.
- [ ] PWA passes Lighthouse PWA audit score >= 90.
- [ ] Dashboard is fully operable in portrait mode on a 375px-wide viewport (iPhone SE
      reference size).

---

## Dependencies

- Pi Camera Module physically attached and enabled via raspi-config before any streaming
  work can be validated end-to-end.
- USB microphone or audio HAT must be attached for audio streaming and cry detection.
- ML API contract (Unix socket schema) must be finalized before backend wires up
  alert forwarding.
- Backend WebSocket `/ws/alerts` endpoint must be live before frontend can integrate
  real alerts.
- Backend streaming endpoints must be live before frontend can display the feed.

---

## Open Questions

1. Should motion detection operate on full-frame diff or a configurable region-of-interest
   (e.g., crib bounding box)? ROI would reduce false positives from shadows/pets but
   requires a calibration step.
2. What is the acceptable end-to-end alert latency budget? Current target is 500ms
   ML-to-browser; is this sufficient for the safety use case?
3. Should the alert history persist across server restarts (SQLite) or is in-memory
   (lost on restart) acceptable for v1?
4. Is MJPEG sufficient for Phase 1, or do we need WebRTC for lower latency? MJPEG is
   simpler to implement but typically adds 200–500ms over WebRTC.
5. Do we need authentication (PIN or password) on the stream endpoints, given the system
   is LAN-only?
