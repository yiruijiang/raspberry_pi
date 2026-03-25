"""
evaluate_model.py — Offline precision/recall evaluation for the cry detection model.

Runs the TFLite model over a directory of labelled WAV files and prints a
classification report.  Used to verify the >= 85% precision acceptance criterion
before deploying a new model version.

Directory layout expected
-------------------------
    test_dir/
        cry/        *.wav files labelled as infant cry
        non_cry/    *.wav files labelled as non-cry (background noise, speech, etc.)

Usage
-----
    python src/ml/utils/evaluate_model.py \\
        --model src/ml/models/cry_detection.tflite \\
        --test-dir data/test/ \\
        [--threshold 0.75] \\
        [--sample-rate 16000]

Output
------
Prints precision, recall, F1, and confusion matrix to stdout.
Writes a JSON results file to the same directory as the model.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Allow running as a script from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AudioConfig, ModelConfig
from cry_detector import CryDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _load_wav(path: str, target_sr: int) -> np.ndarray:
    """Load a WAV file and resample to ``target_sr`` if needed.

    Parameters
    ----------
    path:
        Path to the WAV file.
    target_sr:
        Target sample rate in Hz.

    Returns
    -------
    numpy.ndarray
        Mono float32 array, values in ``[-1, 1]``.
    """
    import librosa
    audio, _ = librosa.load(path, sr=target_sr, mono=True, dtype=np.float32)
    return audio


def _clip_to_window(audio: np.ndarray, window_frames: int) -> List[np.ndarray]:
    """Split audio into non-overlapping windows of ``window_frames`` samples.

    Short clips are zero-padded to exactly one window.

    Parameters
    ----------
    audio:
        1-D float32 array.
    window_frames:
        Number of samples per window.

    Returns
    -------
    list of numpy.ndarray
        Each element has shape ``(window_frames,)``.
    """
    windows = []
    if len(audio) < window_frames:
        padded = np.zeros(window_frames, dtype=np.float32)
        padded[: len(audio)] = audio
        windows.append(padded)
    else:
        n_windows = len(audio) // window_frames
        for i in range(n_windows):
            windows.append(audio[i * window_frames : (i + 1) * window_frames])
    return windows


def evaluate(
    model_path: str,
    test_dir: str,
    threshold: float = 0.75,
    sample_rate: int = 16_000,
) -> dict:
    """Run evaluation and return a results dict.

    Parameters
    ----------
    model_path:
        Path to the TFLite model file.
    test_dir:
        Root directory containing ``cry/`` and ``non_cry/`` subdirectories.
    threshold:
        Confidence threshold used to decide cry vs. silence.
    sample_rate:
        Sample rate to resample all audio to.

    Returns
    -------
    dict
        Keys: ``precision``, ``recall``, ``f1``, ``accuracy``,
        ``true_positives``, ``false_positives``, ``true_negatives``,
        ``false_negatives``, ``n_cry_clips``, ``n_non_cry_clips``.
    """
    audio_cfg = AudioConfig(sample_rate=sample_rate)
    model_cfg = ModelConfig(model_path=model_path, cry_threshold=threshold)
    detector = CryDetector(audio_cfg, model_cfg)

    cry_dir = os.path.join(test_dir, "cry")
    non_cry_dir = os.path.join(test_dir, "non_cry")

    if not os.path.isdir(cry_dir) or not os.path.isdir(non_cry_dir):
        raise FileNotFoundError(
            f"Expected subdirectories 'cry/' and 'non_cry/' inside '{test_dir}'"
        )

    def _collect_wavs(directory: str) -> List[str]:
        return [
            str(p)
            for p in Path(directory).rglob("*.wav")
        ]

    cry_files = _collect_wavs(cry_dir)
    non_cry_files = _collect_wavs(non_cry_dir)
    logger.info("Found %d cry files, %d non-cry files", len(cry_files), len(non_cry_files))

    tp = fp = tn = fn = 0

    def _evaluate_file(path: str, true_label: int) -> Tuple[int, int]:
        """Return (correct, total) window counts for a single file."""
        nonlocal tp, fp, tn, fn
        audio = _load_wav(path, sample_rate)
        windows = _clip_to_window(audio, audio_cfg.window_frames)
        for window in windows:
            event = detector.infer(window)
            predicted = 1 if event.event_type == "cry" else 0
            if true_label == 1 and predicted == 1:
                tp += 1
            elif true_label == 0 and predicted == 1:
                fp += 1
            elif true_label == 0 and predicted == 0:
                tn += 1
            elif true_label == 1 and predicted == 0:
                fn += 1

    for path in cry_files:
        _evaluate_file(path, true_label=1)

    for path in non_cry_files:
        _evaluate_file(path, true_label=0)

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / total if total > 0 else 0.0

    results = {
        "model_path": model_path,
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "n_cry_clips": len(cry_files),
        "n_non_cry_clips": len(non_cry_files),
        "n_windows_evaluated": total,
    }

    _print_report(results)
    return results


def _print_report(r: dict) -> None:
    print("\n" + "=" * 60)
    print("CRY DETECTION MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Model:     {r['model_path']}")
    print(f"  Threshold: {r['threshold']:.2f}")
    print(f"  Clips:     {r['n_cry_clips']} cry / {r['n_non_cry_clips']} non-cry")
    print(f"  Windows:   {r['n_windows_evaluated']}")
    print()
    print(f"  Precision: {r['precision']:.4f}  (spec target >= 0.85)")
    print(f"  Recall:    {r['recall']:.4f}")
    print(f"  F1 Score:  {r['f1']:.4f}")
    print(f"  Accuracy:  {r['accuracy']:.4f}")
    print()
    print("  Confusion matrix:")
    print(f"    TP={r['true_positives']:5d}  FP={r['false_positives']:5d}")
    print(f"    FN={r['false_negatives']:5d}  TN={r['true_negatives']:5d}")
    passed = r['precision'] >= 0.85
    print()
    print(f"  Spec gate (precision >= 85%): {'PASS' if passed else 'FAIL'}")
    print("=" * 60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate cry detection TFLite model on a labelled test set"
    )
    parser.add_argument("--model", required=True, help="Path to .tflite model file")
    parser.add_argument("--test-dir", required=True, help="Root of labelled test dataset")
    parser.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Audio sample rate")
    parser.add_argument(
        "--output-json", default=None, help="Optional path to write JSON results"
    )
    args = parser.parse_args()

    results = evaluate(
        model_path=args.model,
        test_dir=args.test_dir,
        threshold=args.threshold,
        sample_rate=args.sample_rate,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results written to %s", args.output_json)

    return 0 if results["precision"] >= 0.85 else 1


if __name__ == "__main__":
    sys.exit(main())
