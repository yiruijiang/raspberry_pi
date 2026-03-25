---
name: Frontend design decisions
description: Key design and architecture decisions made during frontend development
type: project
---

## Styling approach
CSS Modules (*.module.css) + global CSS custom properties in src/styles/global.css.
No Tailwind, no styled-components, no emotion. Chosen for simplicity and LAN-offline compatibility (no CDN deps).

## Color palette (night-friendly)
- --color-bg: #0d0d1a (deep navy)
- --color-surface: #1a1a2e
- --color-text-primary: #e8e4dc (warm white)
- Status: green #2ecc71, amber #f39c12, red #e74c3c, blue #3498db
- Alert cry: #e67e22 (warm orange), alert motion: #9b59b6 (muted purple)

## MJPEG reconnect mechanism
Updating <img src> is sufficient — browser closes old MJPEG connection and opens new one.
No manual stream teardown needed. Token refresh at 50s triggers src update, which causes natural reconnect.

## iOS Safari MJPEG risk — graceful degradation approach (2026-03-25)
VideoStream.tsx now detects iOS at runtime (UA + maxTouchPoints for iPadOS desktop mode) and
renders a non-blocking advisory banner: "Live video may not be supported on iOS Safari. If the
stream doesn't appear, please use Chrome or the desktop browser."
Banner uses muted amber tones (rgba(80,60,10,0.55) bg + rgba(220,180,50,0.9) icon) — night-safe.
A real-device test on iOS Safari 16+ is still REQUIRED before production release.
If confirmed broken, escalate to PM as scope change (WebRTC or HLS fallback). Cannot fix frontend-only.

## WebSocket auth
JWT passed as ?token= query param (browser WS API cannot set custom headers).
This matches openapi.yaml /ws/alerts spec.

## Error boundaries
Three separate ErrorBoundary instances wrap: video feed section, alert panel section.
One crash does not take down the whole dashboard.

## PWA service worker
Vite PWA plugin (Workbox) caches app shell. API and stream URLs are excluded from cache
(navigateFallbackDenylist: /v1/, /ws/). Theme color #1a1a2e.

**Why:** Night-friendly UX is a core product requirement from baby_monitor_spec.md.
