# ADR-002: TFLite Model Source for Cry Detection

**Date**: 2026-03-25
**Status**: Accepted
**Author**: product-manager
**Deciders**: product-manager

---

## Context

Phase 2 is blocked. The ml-engineer has fully scaffolded the cry detection pipeline
in `src/ml/` and is waiting on `src/ml/models/cry_detection.tflite` before any
end-to-end testing or integration work can proceed. A sourcing decision is required
immediately.

Two options were evaluated:

**Option A — YAMNet (pre-trained, fine-tuned)**
YAMNet is a Google audio classification model trained on AudioSet (521 classes,
including baby cry at class index 20). It is available as a TFLite export from
TensorFlow Hub. The ml-engineer's README documents a fine-tune path using YAMNet
as a feature extractor with a lightweight downstream classifier trained on
ESC-50 / Donate-a-cry data.

**Option B — Custom model trained from scratch**
Binary CNN classifier trained on labelled baby cry data (ESC-50 + Donate-a-cry
augmentation). Higher ceiling on precision/recall for the specific task.
Requires dataset sourcing, training infrastructure, and 1–3 weeks of additional
lead time.

---

## Decision

**Use Option A: YAMNet fine-tuned with a lightweight downstream classifier.**

Fine-tune using YAMNet embeddings + a small Dense head trained on ESC-50 class 38
("crying baby") and the Donate-a-cry corpus. Export to INT8 TFLite.

---

## Rationale

1. **Phase 2 is already delayed.** Option B adds weeks of dataset collection and
   training infrastructure work before a single line of pipeline code can be
   validated. That delay compounds downstream: TASK-203 (backend alert ingestion)
   and TASK-204 (frontend alert UI) cannot start until the ML socket contract is
   proven out on real output.

2. **Acceptance criteria are achievable with YAMNet.** The spec requires precision
   >= 0.90, recall >= 0.85, latency < 200 ms on RPi. YAMNet INT8 is documented to
   run well under 200 ms on RPi 4. The precision/recall bar is high but reachable
   with fine-tuning on domain-specific data (Donate-a-cry corpus, ~1,000 labelled
   clips). If fine-tuning falls short, the custom model path (Option B) remains
   open as a v2 upgrade without changing the pipeline interface.

3. **Privacy constraint is satisfied.** YAMNet runs fully on-device. No audio
   leaves the LAN. The model artifact is a static file — no cloud dependency at
   inference time.

4. **Custom model is not excluded — it is deferred.** If YAMNet fine-tune fails to
   meet acceptance criteria after a reasonable tuning effort (defined as two
   training iterations), the team will escalate to Option B. That decision should
   be revisited no later than end of Phase 2.

---

## Consequences

- ml-engineer proceeds with YAMNet fine-tune immediately (see TASK-201).
- The pipeline input/output contract (already defined in the ml-engineer's
  scaffold) does not change regardless of which model is used — downstream teams
  are unaffected.
- If precision or recall fails acceptance criteria after fine-tuning, Option B
  becomes the Phase 2 recovery path. This must be flagged explicitly and will
  require a timeline re-plan.
- Model file must not be committed to git (binary artifact). It is placed at
  `src/ml/models/cry_detection.tflite` and referenced via
  `BABY_MONITOR_MODEL_PATH` env var per the ml-engineer's README.

---

## Alternatives Rejected

**Option B (custom from scratch)** — rejected for Phase 2 due to timeline impact.
Dataset collection + training infrastructure + two iterations of tuning represents
a minimum 2–3 week addition to an already-delayed phase. The fine-tune path on
YAMNet provides an adequate starting point with a clear upgrade path.
