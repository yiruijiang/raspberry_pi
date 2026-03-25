# Cry Detector Training Pipeline

This directory contains the YAMNet fine-tune pipeline for baby cry detection,
implemented per TASK-201 and ADR-002.

The training script produces `src/ml/models/cry_detection.tflite` — an INT8-quantised
TFLite model that takes raw 16 kHz mono audio and outputs `[P(not_cry), P(cry)]`.

---

## Quick start (single command)

Run the full pipeline — download data, extract embeddings, train, export:

```bash
cd <repo-root>
pip install -r src/ml/training/requirements.txt
python src/ml/training/train_cry_detector.py
```

The script exits 0 when both acceptance gates pass (precision >= 0.90, recall >= 0.85)
and exits 1 if they do not.  See [Acceptance gates](#acceptance-gates).

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Required by TensorFlow 2.16 |
| 4 GB RAM (dev machine) | YAMNet embedding extraction keeps one batch in memory at a time |
| ~2 GB disk | ESC-50 (~600 MB), Donate-a-cry (~50 MB), processed clips, SavedModel |
| Internet access | For `git clone` of datasets and TF Hub model download (~20 MB) |
| GPU (optional) | CPU-only is fully supported; GPU reduces embedding extraction time |

**Do not run on Raspberry Pi.** Training requires the full TensorFlow package
(~500 MB); the Pi uses `tflite-runtime` for inference only.

---

## Step-by-step instructions

### Step 1 — Set up environment

```bash
cd <repo-root>
python -m venv .venv-training
source .venv-training/bin/activate   # Windows: .venv-training\Scripts\activate
pip install -r src/ml/training/requirements.txt
```

Expected install time: 2–5 minutes depending on network speed.

### Step 2 — Source the Donate-a-cry dataset

The training script will attempt to `git clone` the Donate-a-cry corpus
automatically into `data/raw/baby_cry_detection/`.  If the clone fails (e.g.,
behind a corporate proxy), run it manually:

```bash
mkdir -p data/raw
git clone --depth=1 https://github.com/giulbia/baby_cry_detection.git \
    data/raw/baby_cry_detection
```

The repository contains:
- `data/baby_cry/` — approximately 930 labelled infant cry clips (WAV, various sample rates)
- `data/noise/` — approximately 930 non-cry clips (ambient noise, speech, silence)

**Licence**: MIT.  No additional registration is required.

To add extra labelled data (improves recall on edge cases):
- Additional cry clips: drop WAVs into `data/raw/extra_cry/`
- Additional non-cry clips: drop WAVs into `data/raw/extra_non_cry/`

Any sample rate and channel count is accepted — the pipeline resamples to 16 kHz
mono automatically.

### Step 3 — Source ESC-50

ESC-50 is cloned automatically into `data/raw/ESC-50/`.  If the clone fails:

```bash
git clone --depth=1 https://github.com/karolpiczak/ESC-50.git data/raw/ESC-50
```

Class 38 ("crying_baby") provides 40 clips.  All other classes contribute a
capped number of non-cry clips for the negative class.

### Step 4 — Run training

```bash
python src/ml/training/train_cry_detector.py
```

Optional flags:

```
--repo-root PATH       Repository root (default: auto-detected)
--epochs N             Maximum epochs per attempt (default: 30)
--force-resample       Re-run 16 kHz resampling even if processed files exist
--saved-model-dir DIR  Override SavedModel output dir (default: data/cry_detector_saved/)
--tflite-output PATH   Override TFLite output path
```

What the script does:

1. Clones datasets if absent (ESC-50, Donate-a-cry).
2. Resamples all clips to 16 kHz mono WAV into `data/processed/`.
3. Downloads YAMNet from TF Hub (`https://tfhub.dev/google/yamnet/1`).
4. Extracts mean-pooled 1024-dim YAMNet embeddings for all clips.
5. Splits into 70% train / 15% val / 15% test (stratified, seed=42).
6. Trains: `Input(1024) -> Dense(64, relu) -> Dropout(0.3) -> Dense(2, softmax)`
7. Evaluates on held-out test set.
8. If gates fail, retries once with `Dropout(0.5)`.
9. Exports combined YAMNet + head as a SavedModel.
10. Converts to INT8 TFLite using 200-sample calibration dataset.
11. Writes `src/ml/models/cry_detection.tflite`.

### Step 5 — Check acceptance gates

The script prints a summary and exits non-zero if gates are not met:

```
=================================================================
YAMNet Cry Detector — Training Summary
=================================================================
  Output model:  src/ml/models/cry_detection.tflite
  Model size:    X.XX MB  (spec limit: <= 20 MB)

  Precision:     0.XXXX  (gate >= 0.90) [PASS/FAIL]
  Recall:        0.XXXX  (gate >= 0.85) [PASS/FAIL]
  F1:            0.XXXX

  Acceptance gates: ALL PASSED / FAILED
=================================================================
```

If gates fail after two attempts, escalate to the product-manager per
TASK-201 (the fallback is the custom model path described in
`src/ml/models/README.md`).

---

## Expected runtime

| Phase | Approximate time (CPU-only) |
|---|---|
| Dataset download (first run) | 2–10 min (network dependent) |
| Audio resampling (first run) | 3–8 min (~2000 clips) |
| YAMNet embedding extraction | 10–20 min (depends on clip count) |
| Dense head training (30 epochs) | 1–3 min |
| TFLite INT8 conversion | 2–5 min |
| **Total (first run)** | **~20–40 min** |
| **Subsequent runs (data cached)** | **~15 min** |

With a GPU the embedding extraction phase drops to under 2 minutes.

---

## Output files

| Path | Description |
|---|---|
| `src/ml/models/cry_detection.tflite` | INT8 TFLite model — deploy to Raspberry Pi |
| `src/ml/models/cry_detection_fp32.tflite` | FP32 baseline — for accuracy delta measurement |
| `data/cry_detector_saved/` | TensorFlow SavedModel (intermediate; not deployed) |
| `data/processed/cry/` | 16 kHz mono WAVs — positive class |
| `data/processed/non_cry/` | 16 kHz mono WAVs — negative class |

All `data/` and `*.tflite` paths are excluded from git via `.gitignore`.

---

## Placing the model for inference

After training completes:

```bash
# Version the model file
mv src/ml/models/cry_detection.tflite src/ml/models/cry_detection_v1_int8.tflite

# Create symlink so the pipeline finds it at the default path
ln -sf cry_detection_v1_int8.tflite src/ml/models/cry_detection.tflite
```

Copy to Raspberry Pi:

```bash
scp src/ml/models/cry_detection_v1_int8.tflite pi@<PI_IP>:~/baby_monitor/src/ml/models/
ssh pi@<PI_IP> "cd ~/baby_monitor/src/ml/models && ln -sf cry_detection_v1_int8.tflite cry_detection.tflite"
```

---

## Benchmarking on Raspberry Pi

Run the latency benchmark **on the Pi** (not the dev machine):

```bash
python src/ml/utils/benchmark_model.py \
    --model src/ml/models/cry_detection.tflite \
    --n-runs 200
```

Required results before marking TASK-201 done:
- Latency p95 <= 200 ms
- CPU fraction <= 30%

Record results in `src/ml/models/benchmarks.md`.

---

## Inference pipeline compatibility note

The YAMNet-based model exported by this script accepts a **raw 16 kHz float32
waveform** of arbitrary length (TFLite input shape `[None]`).

The existing `src/ml/cry_detector.py` currently feeds MFCC tensors of shape
`(1, 40, 100, 1)` to the model.  Before running the pipeline end-to-end, the
inference code must be updated to pass the raw PCM window directly to the model
(bypassing the MFCC preprocessor for this model variant).

The socket output contract `{"type", "confidence", "timestamp"}` is unchanged —
the backend-engineer does not need to wait for this adaptation.

---

## Acceptance gates

| Gate | Spec | Script behaviour |
|---|---|---|
| Precision | >= 0.90 | Exit code 1 if not met |
| Recall | >= 0.85 | Exit code 1 if not met |
| Model size | <= 20 MB | Warning logged if exceeded |
| Latency p95 | <= 200 ms (on RPi) | Checked by `benchmark_model.py` |
| CPU fraction | <= 30% (on RPi) | Checked by `benchmark_model.py` |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'tensorflow'`**
Install the training requirements: `pip install -r src/ml/training/requirements.txt`.
Do not confuse with the inference `src/ml/requirements.txt`.

**`git clone` fails for datasets**
Clone manually — see Step 2 and Step 3 above.

**TFLite conversion fails with `RESOURCE_EXHAUSTED`**
Reduce `CALIBRATION_SAMPLES` in `train_cry_detector.py` to 100.

**Gates fail after two attempts**
Escalate to product-manager per TASK-201.  The fallback is the custom model
path (Option B) described in `src/ml/models/README.md`.
