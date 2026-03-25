---
name: ML socket contract
description: Agreed interface between ml-engineer (writer) and backend-engineer (reader) for alert events
type: project
---

Protocol: Unix domain socket, newline-delimited JSON (one event per line, UTF-8).

Default socket path: `/tmp/baby_monitor_ml.sock` — configurable via `ML_SOCKET_PATH` env var. This path must match in both `src/ml/` and `src/backend/.env`.

**Event schema:**
```json
{
  "type":       "cry" | "motion",
  "confidence": 0.0 to 1.0,
  "timestamp":  "<ISO 8601 UTC string>"
}
```

Validation: backend enforces type ∈ {cry, motion}, confidence is numeric, timestamp is string. Malformed events are logged and skipped — they do not crash the reader.

**Why:** Spec (baby_monitor_spec.md §Communication Contracts) defines this; backend forwards events as-is to WebSocket clients.

**How to apply:** If the ml-engineer proposes any schema change (additional fields, different type values), the backend `_is_valid_alert()` function and the OpenAPI `AlertEvent` schema must both be updated. Coordinate before Phase 2 integration.

Status as of 2026-03-25: schema is spec-defined but not yet validated end-to-end with running ML pipeline. Backend reader armed and retrying — will auto-connect when ML socket appears.
