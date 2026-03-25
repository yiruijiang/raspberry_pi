# Cry Detection Model — Benchmarks

See `src/ml/models/README.md` for model setup instructions.
See `src/ml/training/README.md` for training instructions.

---

## v1 — YAMNet fine-tune, INT8 (TASK-201)

**Status**: Template — fill in after running `benchmark_model.py` on Raspberry Pi 4.

Training script: `src/ml/training/train_cry_detector.py`
Trained: *(fill in date after training run)*

### Model characteristics

| Field | Value |
|---|---|
| Model file | `cry_detection_v1_int8.tflite` |
| Architecture | YAMNet (frozen) + Dense(64, relu) + Dropout(0.3) + Dense(2, softmax) |
| Feature input | Raw 16 kHz mono float32 waveform (variable length) |
| Internal features | Mean-pooled YAMNet embeddings, 1024-dim |
| Model output | `(1, 2)` — `[P(not_cry), P(cry)]` |
| Quantisation | INT8 post-training (float32 I/O preserved) |
| Training data | ESC-50 class 38 + Donate-a-cry corpus |
| Train split | 70% train / 15% val / 15% test, seed=42 |

### Accuracy (dev machine, held-out test set)

| Metric | Value | Spec gate |
|---|---|---|
| Precision | — | >= 0.90 |
| Recall | — | >= 0.85 |
| F1 | — | — |
| Accuracy delta vs FP32 | —% | <= 3% |

*(Fill in after running `train_cry_detector.py` — values printed in training summary)*

### RPi 4 benchmark

Run on Raspberry Pi 4B after copying model with:

```bash
python src/ml/utils/benchmark_model.py \
    --model src/ml/models/cry_detection.tflite \
    --n-runs 200 \
    --output-json src/ml/models/benchmark_v1_results.json
```

| Field | Value | Spec gate |
|---|---|---|
| RPi hardware | Raspberry Pi 4B | — |
| RAM (OS) | — GB | — |
| Python | 3.11 | — |
| tflite-runtime | 2.14.0 | — |
| Window duration | 1.0 s | — |
| Sample rate | 16 000 Hz | — |
| Quantisation | INT8 post-training | — |
| Model size (disk) | — MB | <= 20 MB |
| Model RAM footprint | — MB | <= 100 MB |
| Inference latency p50 | — ms | — |
| Inference latency p95 | — ms | <= 200 ms |
| Inference latency p99 | — ms | — |
| Inference CPU fraction | —% | <= 30% |
| FP rate (10 min ambient) | — events | < 2 |
| Date benchmarked | — | — |

*(Fill in after running benchmark on RPi)*

### Spec acceptance gates

- [ ] Precision >= 0.90 on held-out test set
- [ ] Recall >= 0.85 on held-out test set
- [ ] Accuracy delta vs FP32 <= 3%
- [ ] Model file size <= 20 MB on disk
- [ ] Model RAM footprint <= 100 MB
- [ ] Inference latency p95 <= 200 ms on Raspberry Pi 4
- [ ] CPU utilisation <= 30% on a single core during continuous inference
- [ ] False positive rate < 2 events per 10 minutes of typical ambient audio
- [ ] Pipeline starts and begins inference within 10 seconds of process start
- [ ] Symlink `cry_detection.tflite` -> `cry_detection_v1_int8.tflite` in place

---

## Adding future model versions

Copy the v1 template above, increment the version, and fill in the new numbers.
Keep all historical versions in this file for regression tracking.
