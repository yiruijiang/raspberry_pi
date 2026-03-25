---
name: Phase Plan
description: Three-phase delivery order, task IDs, owners, and current completion status as of spec creation
type: project
---

Spec written 2026-03-25. All phases are Backlog as of that date.

**Why:** Phases are sequential by hard dependency. Phase 1 streaming must work before
ML can be validated end-to-end. ML socket contract must be stable before backend alert
forwarding is built.

**How to apply:** Never dispatch a downstream phase until the upstream phase exit
criteria are met. Within a phase, dispatch parallel tasks simultaneously.

## Phase 1 — Streaming (entry: hardware attached)
- TASK-101 backend-engineer Video stream endpoint [PARALLEL]
- TASK-102 backend-engineer Audio stream endpoint [PARALLEL]
- TASK-103 frontend-engineer Feed display + reconnect [after 101+102]
- TASK-104 frontend-engineer Stream health indicator [after 103]

## Phase 2 — ML Alerts (entry: Phase 1 done)
- TASK-201 ml-engineer Cry detection pipeline [PARALLEL] — CRITICAL PATH, Large
- TASK-202 ml-engineer Motion detection pipeline [PARALLEL]
- TASK-203 backend-engineer Alert ingestion + WS broadcast [after 201+202 schema finalized]
- TASK-204 frontend-engineer Alert feed UI [after 203]
- TASK-205 frontend-engineer Threshold settings [after 204]

## Phase 3 — UI Polish (entry: Phase 2 done)
- TASK-301 frontend-engineer Dashboard layout [PARALLEL]
- TASK-302 frontend-engineer Night mode [PARALLEL]
- TASK-303 frontend-engineer PWA packaging [after 301]
- TASK-304 frontend-engineer Alert history log [PARALLEL]

Critical path: 101→103→[P1]→201→203→204→[P2]→301→303→[P3 done]
Longest pole: TASK-201 (cry detection, Large). Start immediately when Phase 2 opens.

## Open Questions (unresolved at spec time)
1. ROI for motion detection vs full-frame diff
2. 500ms ML-to-browser latency — sufficient for safety use case?
3. Alert history persistence: SQLite vs in-memory for v1
4. MJPEG vs WebRTC for Phase 1 video
5. Authentication on stream endpoints (LAN-only but still open)
