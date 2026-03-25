# TASK-201: TFLite Model Setup — Cry Detection (YAMNet Fine-Tune)

**Date**: 2026-03-25
**Owner**: ml-engineer
**Spec**: docs/specs/decisions/ADR-002-tflite-model-source.md
**Status**: In Progress
**Complexity**: Large

---

## Description

Produce `src/ml/models/cry_detection.tflite` (INT8 quantised) using YAMNet as a
frozen feature extractor with a fine-tuned lightweight downstream classifier.
This file is the sole blocker for Phase 2. Once the file is placed and passes
acceptance criteria, TASK-203 (backend) and TASK-202 (motion detection) can
proceed in parallel.

**Decision context:** ADR-002 chose YAMNet fine-tune over a custom model from
scratch. The pipeline scaffold is already in place. This task is only about
acquiring, fine-tuning, quantising, validating, and placing the model file.

---

## Dependencies

- Blocked by: ADR-002 (accepted — this task may now proceed)
- Blocks: TASK-203 (backend alert ingestion) — blocked until model produces
  valid output on the established socket contract
- Blocks: end-to-end Phase 2 validation

---

## Step-by-Step Instructions

### Step 1 — Set up environment (dev machine, not RPi)

Training and conversion must be done on a machine with TensorFlow >= 2.13.
The RPi is for inference only.

```
pip install tensorflow tensorflow-hub soundfile librosa numpy scikit-learn
```

### Step 2 — Download YAMNet from TensorFlow Hub

```python
import tensorflow_hub as hub

# Download once; hub caches locally
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
```

YAMNet expects mono 16 kHz float32 audio. It returns:
- `scores`: shape (N, 521) — per-frame class probabilities
- `embeddings`: shape (N, 1024) — per-frame feature embeddings
- `log_mel_spectrogram`: shape (N, 64, 96)

Use `embeddings` as input to the downstream classifier.
Baby cry class index in raw YAMNet scores is **20**.

### Step 3 — Source training data

Download both datasets:

**ESC-50 (class 38 = "crying_baby"):**
```
git clone https://github.com/karolpiczak/ESC-50.git
# Audio in ESC-50/audio/, labels in ESC-50/meta/esc50.csv
# Filter rows where target == 38
```

**Donate-a-cry corpus:**
```
git clone https://github.com/giulbia/baby_cry_detection.git
# Labelled cry clips in data/baby_cry/, non-cry in data/noise/
```

Combine into a two-class dataset: `cry` and `non_cry`.
Target minimum: 500 clips per class. Augment if short (time stretch +-10%,
additive noise SNR 10-20 dB).

Resample all audio to 16 kHz mono WAV before processing.

### Step 4 — Extract YAMNet embeddings

For each audio clip, extract the mean-pooled embedding (shape 1024) using
the YAMNet embeddings output. This is the feature vector for the classifier.

```python
import numpy as np
import soundfile as sf

def extract_embedding(wav_path, yamnet_model):
    audio, sr = sf.read(wav_path)
    assert sr == 16000, "Resample to 16 kHz first"
    audio = audio.astype(np.float32)
    scores, embeddings, _ = yamnet_model(audio)
    # Mean-pool across frames -> shape (1024,)
    return np.mean(embeddings.numpy(), axis=0)
```

Run this for all clips. Save as numpy arrays with labels (0 = non_cry, 1 = cry).

### Step 5 — Train the downstream classifier

A small Dense head on top of frozen YAMNet embeddings:

```python
import tensorflow as tf

inputs = tf.keras.Input(shape=(1024,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

classifier = tf.keras.Model(inputs, outputs)
classifier.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
classifier.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
)
classifier.save('cry_classifier_keras/')
```

Target on validation set before export: accuracy >= 92%, recall on cry class >= 87%.
If not met after 30 epochs, increase dropout to 0.5 and re-run one more iteration.

### Step 6 — Build and export the combined TFLite model

The final TFLite model must be a single artifact that takes raw 16 kHz mono
audio and outputs [not_cry_prob, cry_prob]. Wrap YAMNet + classifier into one
SavedModel, then convert.

```python
class CryDetector(tf.Module):
    def __init__(self, yamnet, classifier):
        self.yamnet = yamnet
        self.classifier = classifier

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='waveform')
    ])
    def __call__(self, waveform):
        _, embeddings, _ = self.yamnet(waveform)
        pooled = tf.reduce_mean(embeddings, axis=0, keepdims=True)
        return self.classifier(pooled)

combined = CryDetector(yamnet_model, classifier)
tf.saved_model.save(combined, 'cry_detector_saved/')
```

Export to INT8 TFLite:

```python
def representative_dataset():
    for sample in calibration_samples:   # 100-500 float32 waveforms
        yield [sample]

converter = tf.lite.TFLiteConverter.from_saved_model('cry_detector_saved/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32   # keep float input for audio
converter.inference_output_type = tf.float32  # keep float output for probs

tflite_model = converter.convert()

with open('src/ml/models/cry_detection.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Model size: {len(tflite_model) / 1024:.1f} KB")
```

Model size must be <= 20 MB. INT8 YAMNet + small head is expected to be well
under this limit.

### Step 7 — Validate on the dev machine

Run the evaluation script against a held-out test split (not used in training):

```
python src/ml/utils/evaluate_model.py \
  --model src/ml/models/cry_detection.tflite \
  --test-dir data/test/
```

Required before moving to RPi: precision >= 0.90, recall >= 0.85.

### Step 8 — Benchmark on the Raspberry Pi

Copy the model file to the Pi and run the latency benchmark ON the Pi:

```
scp src/ml/models/cry_detection.tflite pi@<PI_IP>:~/baby_monitor/src/ml/models/

# SSH into the Pi:
python src/ml/utils/benchmark_model.py \
  --model src/ml/models/cry_detection.tflite \
  --n-runs 200
```

Required: p95 latency <= 200 ms, CPU utilisation <= 30% on a single core.

### Step 9 — Document benchmark results

Record all results in `src/ml/models/benchmarks.md` (create if absent).
Include: date, model version, dataset split sizes, precision, recall, p95 latency,
CPU utilisation, model file size, and accuracy delta vs FP32 baseline (<= 3%).

### Step 10 — Version and symlink

Name the file with a version suffix and create a symlink:

```
mv src/ml/models/cry_detection.tflite src/ml/models/cry_detection_v1_int8.tflite
ln -sf cry_detection_v1_int8.tflite src/ml/models/cry_detection.tflite
```

---

## Acceptance Criteria (from spec)

- [ ] Precision >= 0.90 on held-out test set
- [ ] Recall >= 0.85 on held-out test set
- [ ] False positive rate < 2 events per 10 minutes of typical ambient household audio
- [ ] Model file size <= 20 MB on disk
- [ ] Inference latency p95 <= 200 ms on Raspberry Pi 4
- [ ] CPU utilisation <= 30% on a single core during continuous inference
- [ ] Model loads and begins inference within 10 seconds of process start
- [ ] Accuracy delta vs FP32 baseline <= 3% (documented in benchmarks.md)
- [ ] Benchmark results recorded in src/ml/models/benchmarks.md
- [ ] symlink cry_detection.tflite -> cry_detection_v1_int8.tflite in place

---

## Definition of Done

All acceptance criteria above are checked. The ml-engineer notifies the
product-manager and backend-engineer that TASK-201 is complete, so TASK-203
can be dispatched.

---

## Blockers

**Current blocker (as of 2026-03-25):** None — ADR-002 is now accepted and
this task is unblocked. Proceed immediately.

**Escalation path:** If fine-tuned YAMNet fails to meet precision/recall after
two training iterations, escalate to product-manager. The fallback is the custom
model path (Option B in src/ml/models/README.md). This will require a Phase 2
timeline re-plan.

---

## Notes

- Do not commit cry_detection.tflite to git. It is a binary artifact. Confirm
  it is covered by .gitignore before placing the file.
- The environment variable BABY_MONITOR_MODEL_PATH overrides the default path
  if you need to test multiple model versions side-by-side.
- Training must be done on a dev machine. Do not attempt to train on the RPi.
