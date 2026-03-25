# Cry Detection Model — Benchmarks

**Status**: Pending — no model file present yet.
Run `src/ml/utils/benchmark_model.py` on the Raspberry Pi once `cry_detection.tflite`
is in place and record results here.

See `src/ml/models/README.md` for instructions on obtaining or training the model.

---

## Benchmark template

Fill in after running on hardware:

| Field | Value |
|---|---|
| Model file | `cry_detection_vX_int8.tflite` |
| Model size (disk) | — MB |
| RPi hardware | Raspberry Pi 4B |
| RAM (OS) | — GB |
| Python | 3.11 |
| tflite-runtime | 2.14.0 |
| Window duration | 1.0 s |
| Sample rate | 16 000 Hz |
| n_mfcc | 40 |
| Quantisation | INT8 post-training |
| Inference latency p50 | — ms |
| Inference latency p95 | — ms (spec target: <= 200 ms) |
| Inference latency p99 | — ms |
| Inference CPU fraction | —% (spec target: <= 30%) |
| Model RAM footprint | — MB (spec target: <= 100 MB) |
| Precision (held-out set) | —% (spec target: >= 85%) |
| Recall (held-out set) | —% |
| F1 (held-out set) | —% |
| FP rate (10 min ambient) | — events (spec target: < 2) |
| Accuracy delta vs FP32 | —% (spec target: <= 3%) |
| Date benchmarked | — |

---

## Spec acceptance gates

- [ ] Inference latency p95 <= 200 ms
- [ ] CPU utilisation <= 30% on a single core during continuous inference
- [ ] Model file size <= 20 MB on disk
- [ ] Model RAM footprint <= 100 MB
- [ ] Precision >= 85% on held-out cry / non-cry test set
- [ ] False positive rate < 2 events per 10 minutes of typical ambient audio
- [ ] Accuracy delta vs. FP32 baseline <= 3%
- [ ] Pipeline starts and begins inference within 10 seconds of process launch
