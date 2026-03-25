---
name: Project context
description: Baby monitor project overview, current phase, team ownership, and key dependencies
type: project
---

Raspberry Pi baby monitor: self-hosted, LAN-only, no cloud.
Pi 4B (2GB+), Pi Camera Module v2, USB microphone.
Python 3.11, FastAPI backend, React 18 TypeScript frontend, TFLite ML pipeline.

Phase 1 (streaming) implemented 2026-03-25. Phase 2 (ML alerts) wired but not yet connected.

**Why:** Parents need low-latency, privacy-preserving monitor without vendor lock-in.

**How to apply:** All new features must stay within the LAN-only, single-household scope until PM explicitly expands it. Do not introduce cloud dependencies.

Current phase gate:
- Phase 1 backend (TASK-101, TASK-102) implemented — frontend-engineer (TASK-103) is now unblocked.
- Phase 2 backend (TASK-203) is implemented but blocked on ml-engineer confirming the Unix socket is live.
- Phase 3 not started.

Key dependency: `/v1/stream/video` uses Bearer JWT, but browsers cannot send Authorization headers on `<img>` tag requests. This needs resolution with the frontend-engineer before TASK-103 can complete. Options: short-lived signed URL or session cookie fallback.
