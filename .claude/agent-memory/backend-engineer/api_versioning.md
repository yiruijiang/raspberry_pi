---
name: API versioning and contract rules
description: URL prefix policy, OpenAPI spec location, and breaking-change coordination rules
type: project
---

All REST endpoints use `/v1/` prefix. WebSocket endpoint is `/ws/alerts` (no version prefix — versioned via query param if needed in future).

OpenAPI 3.1 spec: `src/backend/docs/openapi.yaml` — this is the authoritative API contract. Update immediately when any endpoint changes.

Backend design doc: `docs/design/backend-design.md` — update the API change log table for every endpoint addition or modification.

**Breaking change policy:** Coordinate with frontend-engineer before any change that modifies request/response schemas, removes endpoints, or changes auth requirements. Do not break the contract silently.

**Non-breaking additions** (new optional fields in responses, new endpoints) can be shipped without frontend coordination but must still update the OpenAPI spec.

**Known open issue (2026-03-25):** Browser `<img>` tags cannot send Authorization headers, which conflicts with JWT protection on `/v1/stream/video`. This must be resolved with the frontend-engineer before TASK-103. Options: short-lived signed URL parameter, or session cookie with SameSite=Strict.
