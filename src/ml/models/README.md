# cry_detection.tflite — Model Setup Guide

This directory holds the TFLite model file(s) consumed by the cry detection
pipeline.  The model file is **not committed to the repository** because it is
a binary artifact that may be too large for git and must be generated from a
training run or sourced from a pre-trained checkpoint.

---

## Expected file

| File | Purpose |
|---|---|
| `cry_detection.tflite` | INT8 quantised TFLite model for on-device inference |
| `cry_detection_fp32.tflite` | FP32 baseline (optional, for accuracy comparison) |

Path is configurable via `BABY_MONITOR_MODEL_PATH` environment variable.
Default: `src/ml/models/cry_detection.tflite`

---

## Option A — Use a pre-trained checkpoint (recommended for quick start)

The following open datasets include labelled infant cry audio and can be used
to fine-tune or directly use a pre-trained model:

| Dataset | URL | Notes |
|---|---|---|
| ESC-50 | https://github.com/karolpiczak/ESC-50 | 50 environmental sound classes; "crying baby" is class 38 |
| Donate-a-cry corpus | https://github.com/giulbia/baby_cry_detection | ~1,000 labelled clips of infant cry vs. noise |
| AudioSet (Google) | https://research.google.com/audioset/ | Large-scale; "crying" ontology node `/m/01jg02` |

Pre-trained MobileNet V2 audio classifiers fine-tuned on ESC-50 or AudioSet
are available via TensorFlow Hub:

```
https://tfhub.dev/google/yamnet/1
```

YAMNet produces 521-class embeddings; the "crying" class index is 20.
It can be used directly (thresholding class 20 probability) or as a feature
extractor for a lightweight downstream classifier.

---

## Option B — Train a custom model from scratch

### 1. Prepare data

Recommended directory layout:

```
data/
  train/
    cry/       # *.wav, 16 kHz mono, any duration
    non_cry/   # background, speech, white noise, TV, etc.
  val/
    cry/
    non_cry/
```

Collect at least 500 clips per class.  Augment with:
- Time stretching (±10%)
- Pitch shifting (±2 semitones)
- Additive Gaussian noise (SNR 10–20 dB)
- Room impulse response convolution

### 2. Architecture

The pipeline expects a binary classifier (two output logits: [not_cry, cry])
with input shape `(1, 40, T, 1)` where:
- `40` = n_mfcc (configurable in `config.py`)
- `T` = time steps = `ceil(window_frames / hop_length)`
  = `ceil(16000 * 1.0 / 160)` = 100 for the default 1-second window

Recommended architectures (ordered by RPi suitability):
1. **Tiny CNN** — 2–3 Conv2D + GlobalAveragePooling + Dense(2) — ~500 KB FP32
2. **MobileNetV2** (width 0.25x) — ~1 MB FP32, ~250 KB INT8
3. **EfficientNet-Lite0** — ~4 MB FP32, ~1 MB INT8

Avoid transformers or large RNNs — inference latency exceeds 200 ms on RPi 4.

### 3. Training (Keras example skeleton)

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.Input(shape=(40, 100, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(train_ds, validation_data=val_ds, epochs=30)
model.save('cry_detection_keras/')
```

### 4. Export to TFLite (INT8 post-training quantisation)

INT8 quantisation is **required** for production deployment on RPi.  It
reduces model size by ~4x and inference latency by ~2x vs. FP32.

```python
import tensorflow as tf
import numpy as np

# Provide a representative dataset for calibration (100–500 samples).
def representative_dataset():
    for sample in calibration_samples:           # numpy arrays, shape (1, 40, 100, 1)
        yield [sample.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model('cry_detection_keras/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open('src/ml/models/cry_detection.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Model size: {len(tflite_model) / 1024:.1f} KB")
```

### 5. Validate before deploying

Run the acceptance criteria checks:

```bash
# Precision on held-out test set must be >= 85%
python src/ml/utils/evaluate_model.py \
  --model src/ml/models/cry_detection.tflite \
  --test-dir data/test/

# Latency benchmark on RPi (run this ON the Pi, not the dev machine)
python src/ml/utils/benchmark_model.py \
  --model src/ml/models/cry_detection.tflite \
  --n-runs 200
```

---

## Model acceptance criteria (from TASK-201)

Before marking the model as production-ready, all of the following must pass:

- [ ] Precision >= 85% on held-out cry/non-cry test set
- [ ] False positive rate < 2 events per 10 minutes of typical ambient household audio
- [ ] Model file size <= 20 MB on disk
- [ ] Inference latency p95 <= 200 ms on Raspberry Pi 4
- [ ] CPU utilisation <= 30% on a single core during continuous inference
- [ ] Model loads and begins inference within 10 seconds of process start
- [ ] Accuracy delta vs. FP32 baseline <= 3% (document in benchmarks.md)

---

## Model versioning

Name files with a version suffix and update the symlink or `BABY_MONITOR_MODEL_PATH`:

```
models/
  cry_detection_v1_int8.tflite
  cry_detection_v2_int8.tflite   ← current production
  cry_detection.tflite           ← symlink → cry_detection_v2_int8.tflite
```

Record all benchmark results in `src/ml/models/benchmarks.md`.
