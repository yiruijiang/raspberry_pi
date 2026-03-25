---
name: Project Overview
description: Core goals, constraints, and out-of-scope items for the baby monitor v1
type: project
---

Raspberry Pi baby monitor: self-hosted, LAN-only, real-time video/audio streaming with
on-device ML cry and motion detection. Web frontend served as a React PWA.

**Why:** Parents need a privacy-respecting, low-latency, customizable monitor that
filters out false alarms so sleep-deprived caregivers are only alerted when it matters.

**How to apply:** Safety (infant health alerts) always outranks reliability, which
outranks usability, which outranks polish. Use this ordering for any prioritization call.

Key constraints:
- No cloud dependency — everything runs on the Pi on the home network
- No multi-camera support in v1
- No two-way audio, no clip storage, no biometrics beyond cry/motion in v1
- No native iOS/Android app — PWA covers mobile

Tech stack locked:
- ML: Python 3.11 + TensorFlow Lite + OpenCV, on-device inference
- Backend: Python 3.11 + FastAPI, systemd service on Pi
- Frontend: React 18 + TypeScript + Vite, PWA via vite-plugin-pwa
