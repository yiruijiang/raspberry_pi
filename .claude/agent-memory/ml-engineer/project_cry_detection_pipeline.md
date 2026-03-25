---
name: Cry detection pipeline — design, API contract, status
description: Design decisions, API contract, and status for the TASK-201 cry detection audio ML pipeline
type: project
---

Pipeline shipped at `src/ml/` covering TASK-201 (cry detection) from Phase 2 of the baby monitor spec.

**Why:** Spec approved 2026-03-25. Cry detection is the longest-pole task; backend (TASK-203) is blocked until the Unix socket schema is finalised — that contract is now live.

**How to apply:** When resuming work on this pipeline or reviewing backend integration, use the details below as ground truth.

## Unix socket API contract (confirmed, backend may integrate against this)

Wire format: one JSON line per event, newline-terminated.

```json
{"type": "cry", "confidence": 0.9312, "timestamp": "2026-03-25T14:32:01.123Z"}
```

Fields:
- `type`: `"cry"` only emitted when confidence >= threshold; `"silence"` events are NOT sent over the socket (filtered by `cry_detection_pipeline.py`)
- `confidence`: float, 4 decimal places
- `timestamp`: UTC ISO 8601 with millisecond precision, trailing `Z`

Socket path: configurable via `BABY_MONITOR_SOCKET_PATH` env var (default `/tmp/baby_monitor_ml.sock`). Single client (backend process); publisher re-accepts on disconnect.

## Architecture

```
AudioCapture (PyAudio, background thread, deque ring buffer)
    → AudioPreprocessor (librosa MFCCs, shape (1, 40, 100, 1))
    → CryDetector (TFLite INT8, XNNPACK delegate, num_threads=1)
    → SocketPublisher (AF_UNIX SOCK_STREAM, newline-delimited JSON)
```

## Key config defaults (all overridable via env vars)

| Parameter | Default | Env var |
|---|---|---|
| Sample rate | 16 000 Hz | `BABY_MONITOR_SAMPLE_RATE` |
| Window | 1.0 s | `BABY_MONITOR_WINDOW_SECONDS` |
| Overlap | 50% | `BABY_MONITOR_WINDOW_OVERLAP` |
| n_mfcc | 40 | `BABY_MONITOR_N_MFCC` |
| hop_length | 160 | `BABY_MONITOR_HOP_LENGTH` |
| Cry threshold | 0.75 | `BABY_MONITOR_CRY_THRESHOLD` |
| Model path | `src/ml/models/cry_detection.tflite` | `BABY_MONITOR_MODEL_PATH` |

## TASK-201 training pipeline (delivered 2026-03-25)

Training script delivered at `src/ml/training/`. Full pipeline runnable with:

```bash
pip install -r src/ml/training/requirements.txt
python src/ml/training/train_cry_detector.py
```

Files delivered:
- `src/ml/training/train_cry_detector.py` — full YAMNet fine-tune script
- `src/ml/training/data_prep.py` — ESC-50 + Donate-a-cry download, 16 kHz resampling, split utilities
- `src/ml/training/requirements.txt` — training-only deps (TF 2.16.1, tensorflow-hub, librosa, sklearn)
- `src/ml/training/README.md` — step-by-step instructions

## Model input contract — IMPORTANT: pipeline compatibility gap

The YAMNet-based model exported by the training script takes a **raw 16 kHz float32 waveform** (TFLite input shape `[None]`).

The existing `cry_detector.py` currently feeds MFCC tensors of shape `(1, 40, 100, 1)` to the model. These are incompatible.

**Before end-to-end pipeline testing**, `cry_detector.py` must be updated to pass the raw PCM window directly to the YAMNet model (bypassing the MFCC preprocessor). This is a small adaptation task. The socket output contract is unchanged — backend integration can proceed.

## Acceptance criteria status (TASK-201) — as of 2026-03-25

- [ ] Training script run and precision >= 0.90 on held-out test (script written; awaiting human run)
- [ ] Training script run and recall >= 0.85 (script written; awaiting human run)
- [ ] Model file `cry_detection_v1_int8.tflite` produced and symlinked (awaiting run)
- [ ] Inference runs <= 30% CPU on one core (awaiting RPi benchmark)
- [ ] Latency p95 <= 200 ms on RPi (awaiting RPi benchmark)
- [ ] Output JSON written to Unix socket matches schema (implemented)
- [ ] FP rate < 2 events per 10 min ambient audio (awaiting RPi test)
- [ ] Benchmarks recorded in `src/ml/models/benchmarks.md` (template filled; numbers pending)

## Benchmarks

Template in `src/ml/models/benchmarks.md` (v1 row added, numbers pending after RPi run).
