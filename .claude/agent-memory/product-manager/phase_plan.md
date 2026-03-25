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
  — UNBLOCKED 2026-03-25: ADR-002 accepted YAMNet fine-tune path; full step-by-step
    task doc written; ml-engineer may proceed immediately
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

## Phase 1 Task Additions (2026-03-25)
- TASK-105 backend-engineer Stream token endpoint [PARALLEL with TASK-103 dev,
  sequential before TASK-103 integration] — Small — added to unblock TASK-103

## Open Questions (unresolved at spec time)
1. ROI for motion detection vs full-frame diff
2. 500ms ML-to-browser latency — sufficient for safety use case?
3. Alert history persistence: SQLite vs in-memory for v1
4. MJPEG vs WebRTC for Phase 1 video
5. ~~Authentication on stream endpoints~~ RESOLVED: signed URL tokens (ADR-001)

## ADR log
- ADR-001: Video stream auth — signed URL tokens
- ADR-002 (2026-03-25): TFLite model source — YAMNet fine-tune chosen over custom
  model from scratch. Rationale: Phase 2 timeline, achievable acceptance criteria,
  on-device inference. Fallback to custom model if fine-tune fails two iterations.
