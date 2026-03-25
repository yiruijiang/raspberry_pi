# Baby Monitor — Project Context

## Overview
A Raspberry Pi-based baby monitor with computer vision (cry/motion detection),
a streaming backend, and a web/mobile frontend.

## Team Structure
- @product-manager: Owns requirements, roadmap, and task delegation
- @ml-engineer: Owns CV and audio ML pipelines on the Pi
- @backend-engineer: Owns the streaming server and APIs
- @frontend-engineer: Owns the web UI and live feed display

## Sub-Agent Routing Rules

**Parallel dispatch** (when ALL are independent):
- ML model work, backend API work, and UI components with no shared state

**Sequential dispatch** (when dependencies exist):
- Specs must be approved by PM before engineers start
- ML API contract must be defined before frontend consumes it
- Backend streaming endpoints must exist before frontend integrates

## File Ownership
- `src/ml/` → ml-engineer only
- `src/backend/` → backend-engineer only
- `src/frontend/` → frontend-engineer only
- `docs/` → product-manager writes specs; engineers write design docs
