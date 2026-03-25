---
name: Frontend project scaffold — TASK-103
description: Full frontend scaffold for baby monitor PWA created in TASK-103 (2026-03-25)
type: project
---

React 18 + Vite + TypeScript PWA scaffold was created at src/frontend/ for TASK-103 (video stream integration).

**Why:** TASK-103 required a full frontend scaffold — the directory was previously empty.

**How to apply:** All future frontend work builds on this scaffold. Key facts:

- Build toolchain: Vite 5, vite-plugin-pwa (Workbox), @vitejs/plugin-react
- Styling: CSS Modules (*.module.css) + a global CSS custom-properties design token system (src/styles/global.css)
- No CSS-in-JS, no Tailwind — vanilla CSS Modules only
- Auth flow: Bearer JWT from POST /v1/auth/token (HTTP Basic) → held in React state, never localStorage/sessionStorage
- Stream auth: signed URL token from POST /v1/auth/stream-token (TASK-105 dep) → appended as ?stream_token= to <img src>
- Dev proxy: Vite proxies /v1/* and /ws/* to raspberrypi.local:8000 (configured in vite.config.ts)
- VITE_MOCK_STREAM_TOKEN removed (2026-03-25, TASK-103 close) — TASK-105 is deployed; useStreamToken calls real endpoint
- TASK-103 marked complete 2026-03-25: real stream-token endpoint wired, iOS Safari graceful-degradation banner added to VideoStream
