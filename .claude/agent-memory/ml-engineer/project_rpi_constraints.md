---
name: RPi 4 hardware constraints applied in this codebase
description: Decisions made specifically to stay within RPi 4 CPU/RAM/latency limits
type: project
---

Decisions recorded during TASK-201 implementation (2026-03-25).

**Why:** RPi 4B is the target hardware. All ML code must run continuously without thermal throttling or OOM.

**How to apply:** Before adding any new dependency, tensor operation, or model architecture to `src/ml/`, verify it does not violate the constraints below.

## Constraints and applied decisions

| Constraint | Target | Decision made |
|---|---|---|
| RAM | < 100 MB model footprint | INT8 quantised TFLite only; no full TF install on Pi |
| CPU (inference) | <= 30% single core | num_threads=1, XNNPACK delegate, 2 inferences/sec at 50% overlap |
| Inference latency p95 | <= 200 ms | 1s window + 50% overlap; librosa MFCC ~10ms, TFLite INT8 ~20-50ms estimated |
| Model size on disk | <= 20 MB | INT8 quantisation; tiny CNN or MobileNetV2 0.25x recommended |
| Python | 3.11+ | All code uses 3.11+ syntax/dataclasses |

## Packages deliberately chosen for ARM

- `tflite-runtime` (not full `tensorflow`) — 500 MB saved
- `librosa` with pinned `numba` — numpy-based, no GPU ops
- `pyaudio` — uses ALSA directly on Pi OS, no heavy middleware

## Known failure modes to watch for

- `numba` version mismatch with `librosa` breaks JIT on ARM — pin both in requirements.txt
- Batch size > 1 causes memory spikes; always use batch=1 in TFLite input
- `pyaudio.paInt16` is the only format reliably supported across USB mic drivers on Pi
- Full TF package conflicts with `tflite-runtime` — never install both
