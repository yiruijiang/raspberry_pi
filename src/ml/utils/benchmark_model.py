"""
benchmark_model.py — Latency and CPU benchmarking for the cry detection pipeline.

Run this script **on the Raspberry Pi** to measure real inference latency.
Results must be recorded in ``src/ml/models/benchmarks.md`` before marking
the pipeline as production-ready.

Usage
-----
    python src/ml/utils/benchmark_model.py \\
        --model src/ml/models/cry_detection.tflite \\
        [--n-runs 200] \\
        [--sample-rate 16000] \\
        [--window-seconds 1.0]

Output
------
Prints p50, p95, p99, and max inference latency (ms) to stdout.
Prints estimated CPU time fraction (inference time / wall time).
Writes results to stdout and optionally to a JSON file.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AudioConfig, ModelConfig
from cry_detector import CryDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _synthetic_window(n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic audio window resembling a baby cry spectrum."""
    t = np.linspace(0, n_frames / 16000, n_frames, dtype=np.float32)
    # Fundamental ~350 Hz with harmonics, plus light noise.
    signal = (
        0.4 * np.sin(2 * np.pi * 350 * t)
        + 0.2 * np.sin(2 * np.pi * 700 * t)
        + 0.1 * np.sin(2 * np.pi * 1050 * t)
        + 0.05 * rng.standard_normal(n_frames).astype(np.float32)
    )
    return np.clip(signal, -1.0, 1.0)


def run_benchmark(
    model_path: str,
    n_runs: int = 200,
    sample_rate: int = 16_000,
    window_seconds: float = 1.0,
) -> dict:
    """Run ``n_runs`` inferences and collect latency statistics.

    Parameters
    ----------
    model_path:
        Path to the TFLite model file.
    n_runs:
        Number of inference calls.  200 provides stable p95/p99 estimates.
    sample_rate:
        Audio sample rate in Hz.
    window_seconds:
        Length of each synthetic audio window in seconds.

    Returns
    -------
    dict
        Latency percentiles and CPU time fraction.
    """
    audio_cfg = AudioConfig(sample_rate=sample_rate, window_seconds=window_seconds)
    model_cfg = ModelConfig(model_path=model_path)
    detector = CryDetector(audio_cfg, model_cfg)

    rng = np.random.default_rng(seed=42)
    n_frames = audio_cfg.window_frames
    latencies_ms: List[float] = []

    # Warm-up run (JIT, cache warm-up, etc.)
    warmup_window = _synthetic_window(n_frames, rng)
    detector.infer(warmup_window)
    logger.info("Warmup complete — running %d benchmark iterations", n_runs)

    wall_start = time.monotonic()

    for i in range(n_runs):
        window = _synthetic_window(n_frames, rng)
        t0 = time.monotonic()
        detector.infer(window)
        latencies_ms.append((time.monotonic() - t0) * 1000)

    wall_elapsed = (time.monotonic() - wall_start) * 1000  # ms

    arr = np.array(latencies_ms)
    results = {
        "model_path": model_path,
        "n_runs": n_runs,
        "sample_rate": sample_rate,
        "window_seconds": window_seconds,
        "latency_p50_ms": round(float(np.percentile(arr, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(arr, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(arr, 99)), 2),
        "latency_max_ms": round(float(arr.max()), 2),
        "latency_mean_ms": round(float(arr.mean()), 2),
        "total_inference_ms": round(float(arr.sum()), 2),
        "wall_elapsed_ms": round(wall_elapsed, 2),
        # Fraction of wall time spent in inference (rough CPU utilisation proxy).
        "inference_cpu_fraction": round(float(arr.sum()) / wall_elapsed, 4),
    }

    _print_report(results)
    return results


def _print_report(r: dict) -> None:
    print("\n" + "=" * 60)
    print("CRY DETECTION PIPELINE LATENCY BENCHMARK")
    print("=" * 60)
    print(f"  Model:         {r['model_path']}")
    print(f"  Runs:          {r['n_runs']}")
    print(f"  Window:        {r['window_seconds']}s @ {r['sample_rate']} Hz")
    print()
    print(f"  Latency p50:   {r['latency_p50_ms']:.1f} ms")
    print(f"  Latency p95:   {r['latency_p95_ms']:.1f} ms  (spec target <= 200 ms)")
    print(f"  Latency p99:   {r['latency_p99_ms']:.1f} ms")
    print(f"  Latency max:   {r['latency_max_ms']:.1f} ms")
    print(f"  Latency mean:  {r['latency_mean_ms']:.1f} ms")
    print()
    cpu_pct = r["inference_cpu_fraction"] * 100
    print(f"  Inference CPU fraction: {cpu_pct:.1f}%  (spec target <= 30%)")
    p95_pass = r["latency_p95_ms"] <= 200
    cpu_pass = cpu_pct <= 30
    print()
    print(f"  Spec gate (p95 <= 200 ms): {'PASS' if p95_pass else 'FAIL'}")
    print(f"  Spec gate (CPU <= 30%):    {'PASS' if cpu_pass else 'FAIL'}")
    print("=" * 60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark cry detection TFLite model latency on this device"
    )
    parser.add_argument("--model", required=True, help="Path to .tflite model file")
    parser.add_argument("--n-runs", type=int, default=200, help="Number of inference calls")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--output-json", default=None, help="Write JSON results to this path")
    args = parser.parse_args()

    results = run_benchmark(
        model_path=args.model,
        n_runs=args.n_runs,
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results written to %s", args.output_json)

    p95_pass = results["latency_p95_ms"] <= 200
    cpu_pass = results["inference_cpu_fraction"] <= 0.30
    return 0 if (p95_pass and cpu_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
