"""
train_cry_detector.py — YAMNet fine-tune pipeline for baby cry detection.

Overview
--------
This script implements the full training pipeline described in TASK-201 and
ADR-002:

1. Download YAMNet from TensorFlow Hub (cached after first run).
2. Load the ESC-50 class-38 ("crying_baby") clips and Donate-a-cry corpus
   from ``data/`` via ``data_prep.prepare_dataset()``.
3. Extract mean-pooled 1024-dim YAMNet embeddings for every clip.
4. Train a lightweight Dense head:
   ``Input(1024) -> Dense(64, relu) -> Dropout(0.3) -> Dense(2, softmax)``
5. Evaluate precision and recall on a held-out test set.
   Exit code 1 if precision < 0.90 or recall < 0.85 (spec acceptance gates).
6. Save the combined YAMNet + head model as a TensorFlow SavedModel.
7. Export an INT8-quantised TFLite model to ``src/ml/models/cry_detection.tflite``.

Model input / output contract
------------------------------
The exported TFLite model takes a **raw 16 kHz mono float32 waveform** of
arbitrary length and returns a ``(1, 2)`` tensor:

    output[0][0] = P(not_cry)
    output[0][1] = P(cry)

This is a *different input format* from the existing MFCC-based pipeline in
``src/ml/cry_detector.py``.  See the integration note below.

Integration note — inference pipeline compatibility
----------------------------------------------------
The current ``cry_detector.py`` feeds MFCC tensors of shape ``(1, 40, 100, 1)``
to the TFLite model.  The YAMNet-based model produced here takes a raw waveform
(shape ``[None]``).

**To use this model with the existing pipeline**, one of the following must be done
before deploying:

  Option 1 (recommended): Update ``cry_detector.py`` to pass the raw PCM window
  (shape ``[window_frames]``) directly to the model instead of MFCC features.
  The CryDetector._fill_input_buffer path handles float32 inputs — only
  ``_validate_model_shape`` and the ``extract`` call need to be skipped for the
  YAMNet variant.

  Option 2: Wrap the YAMNet model's embedding stage in the preprocessor and
  retrain on MFCC features with a compatible input shape.  This is the Option B
  (custom model from scratch) path and is slower to deliver.

For now the training script produces the YAMNet-based TFLite model exactly as
specified in TASK-201.  Pipeline adaptation is a separate, small task that the
backend-engineer does not need to wait for — the socket output contract is
unchanged.

Hardware constraints
--------------------
- Training must be run on a dev machine with TensorFlow >= 2.13 and at least
  4 GB RAM.  Do NOT run on Raspberry Pi.
- GPU is beneficial but not required; CPU training on ESC-50 + Donate-a-cry
  takes ~15 minutes.

Usage
-----
::

    python src/ml/training/train_cry_detector.py [--repo-root PATH] [--epochs 30]
    python src/ml/training/train_cry_detector.py --help

Exit codes
----------
- 0 : Training completed and both acceptance gates passed
      (precision >= 0.90, recall >= 0.85).
- 1 : Gate failure — retune and rerun.
- 2 : Fatal error (missing data, import error, etc.).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Resolve repository root so this script can be run from any working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent   # src/ml/training -> repo root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spec acceptance gates
# ---------------------------------------------------------------------------
PRECISION_GATE: float = 0.90
RECALL_GATE: float = 0.85

# ---------------------------------------------------------------------------
# Architecture hyper-parameters
# ---------------------------------------------------------------------------
YAMNET_HUB_URL: str = "https://tfhub.dev/google/yamnet/1"
EMBEDDING_DIM: int = 1024          # YAMNet embedding dimension
DENSE_UNITS: int = 64
DROPOUT_RATE: float = 0.3
DROPOUT_RATE_RETRY: float = 0.5    # Used on retry iteration if gates not met
NUM_CLASSES: int = 2
BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-3

# Maximum number of training attempts before giving up.
# On the first failure, we retry once with increased dropout and more epochs.
MAX_TRAINING_ATTEMPTS: int = 2

# Representative dataset size for INT8 calibration
CALIBRATION_SAMPLES: int = 200


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def extract_embeddings(
    file_label_pairs: List[Tuple[str, int]],
    yamnet_model,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract mean-pooled YAMNet embeddings for a list of audio files.

    For each audio clip, YAMNet produces per-frame embeddings of shape
    ``(N_frames, 1024)``.  We take the mean across the frame axis to obtain
    a single ``(1024,)`` vector per clip.  This is the standard approach for
    using YAMNet as a fixed feature extractor.

    Parameters
    ----------
    file_label_pairs:
        List of ``(wav_path, label)`` tuples.  WAV files must be 16 kHz mono
        float32 (as produced by :func:`data_prep.resample_batch`).
    yamnet_model:
        Loaded TensorFlow Hub YAMNet model handle.

    Returns
    -------
    X : numpy.ndarray
        Shape ``(N_clips, 1024)``, dtype float32.
    y : numpy.ndarray
        Shape ``(N_clips,)``, dtype int32.  0 = non_cry, 1 = cry.

    Notes
    -----
    Clips that fail to load or produce zero-length embeddings are skipped with
    a warning.  The function logs a count of skipped clips at the end.
    """
    import tensorflow as tf

    # Import load_wav_16k from sibling module — adjust sys.path if needed.
    sys.path.insert(0, str(_SCRIPT_DIR))
    from data_prep import load_wav_16k

    embeddings_list: List[np.ndarray] = []
    labels_list: List[int] = []
    n_skipped = 0

    total = len(file_label_pairs)
    log_every = max(1, total // 10)

    for i, (wav_path, label) in enumerate(file_label_pairs):
        if i % log_every == 0:
            logger.info("Extracting embeddings: %d / %d", i, total)

        try:
            audio = load_wav_16k(wav_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s — skipping", wav_path, exc)
            n_skipped += 1
            continue

        # YAMNet forward pass — returns (scores, embeddings, log_mel_spectrogram)
        # embeddings shape: (N_frames, 1024)
        try:
            _, frame_embeddings, _ = yamnet_model(
                tf.cast(audio, dtype=tf.float32)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "YAMNet inference failed on %s: %s — skipping", wav_path, exc
            )
            n_skipped += 1
            continue

        emb_np = frame_embeddings.numpy()  # (N_frames, 1024)
        if emb_np.shape[0] == 0:
            logger.warning("Zero-frame embedding for %s — skipping", wav_path)
            n_skipped += 1
            continue

        # Mean-pool across frames -> (1024,)
        pooled = np.mean(emb_np, axis=0)
        embeddings_list.append(pooled)
        labels_list.append(label)

    if n_skipped > 0:
        logger.warning(
            "Skipped %d / %d clips during embedding extraction",
            n_skipped,
            total,
        )

    X = np.stack(embeddings_list, axis=0).astype(np.float32)   # (N, 1024)
    y = np.array(labels_list, dtype=np.int32)                   # (N,)

    logger.info("Embedding extraction complete: X=%s, y=%s", X.shape, y.shape)
    return X, y


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_classifier(dropout_rate: float = DROPOUT_RATE):
    """Build the Dense head that operates on YAMNet embeddings.

    Architecture
    ------------
    ::

        Input(1024)
          -> Dense(64, activation='relu')
          -> Dropout(dropout_rate)
          -> Dense(2, activation='softmax')

    Parameters
    ----------
    dropout_rate:
        Dropout probability applied during training.

    Returns
    -------
    tf.keras.Model
        Uncompiled Keras model.  Call ``.compile()`` before training.
    """
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(EMBEDDING_DIM,), name="yamnet_embedding")
    x = tf.keras.layers.Dense(DENSE_UNITS, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES, activation="softmax", name="output_softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="cry_classifier_head")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    dropout_rate: float = DROPOUT_RATE,
) -> "tf.keras.Model":  # noqa: F821
    """Train the Dense classifier head on pre-computed embeddings.

    Parameters
    ----------
    X_train, y_train:
        Training embeddings and labels.
    X_val, y_val:
        Validation embeddings and labels.
    epochs:
        Maximum number of training epochs.
    dropout_rate:
        Dropout probability.

    Returns
    -------
    tf.keras.Model
        Trained classifier.
    """
    import tensorflow as tf

    model = build_classifier(dropout_rate=dropout_rate)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    logger.info(
        "Training Dense head: %d train, %d val, epochs=%d, dropout=%.2f",
        len(X_train),
        len(X_val),
        epochs,
        dropout_rate,
    )
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_classifier(
    classifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Evaluate precision, recall, and F1 of the trained classifier.

    Parameters
    ----------
    classifier:
        Trained Keras model.
    X_test, y_test:
        Held-out test embeddings and labels.
    threshold:
        Probability threshold for predicting class 1 (cry).

    Returns
    -------
    tuple of (precision, recall, f1)
        All values are floats in ``[0, 1]``.
    """
    probs = classifier.predict(X_test, verbose=0)   # (N, 2)
    y_pred = (probs[:, 1] >= threshold).astype(np.int32)

    tp = int(np.sum((y_pred == 1) & (y_test == 1)))
    fp = int(np.sum((y_pred == 1) & (y_test == 0)))
    fn = int(np.sum((y_pred == 0) & (y_test == 1)))
    tn = int(np.sum((y_pred == 0) & (y_test == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    logger.info(
        "Evaluation on %d test clips:\n"
        "  TP=%-5d  FP=%-5d\n"
        "  FN=%-5d  TN=%-5d\n"
        "  Precision = %.4f  (gate >= %.2f)\n"
        "  Recall    = %.4f  (gate >= %.2f)\n"
        "  F1        = %.4f",
        len(y_test),
        tp, fp, fn, tn,
        precision, PRECISION_GATE,
        recall, RECALL_GATE,
    )

    return precision, recall, f1


# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------


class CryDetectorModule:
    """Combined TensorFlow Module wrapping YAMNet feature extraction and the classifier.

    The exported SavedModel accepts a raw 16 kHz mono float32 waveform of
    arbitrary length and returns a ``(1, 2)`` softmax tensor:
    ``[[P(not_cry), P(cry)]]``.

    Parameters
    ----------
    yamnet:
        Loaded TF Hub YAMNet model.
    classifier:
        Trained Keras Dense-head model.
    """

    def __init__(self, yamnet, classifier) -> None:
        import tensorflow as tf

        # Store as attributes but also wrap in tf.Module so SavedModel
        # captures both sets of variables.
        self._yamnet = yamnet
        self._classifier = classifier

        # Pre-define the concrete function signature for SavedModel export.
        # shape=[None] means variable-length 1-D waveform.
        self._infer = tf.function(
            self._call_impl,
            input_signature=[
                tf.TensorSpec(shape=[None], dtype=tf.float32, name="waveform")
            ],
        )

    def _call_impl(self, waveform):
        """Run YAMNet + classifier on a raw waveform.

        Parameters
        ----------
        waveform:
            1-D float32 tensor, 16 kHz mono.

        Returns
        -------
        tf.Tensor
            Shape ``(1, 2)`` — ``[[P(not_cry), P(cry)]]``.
        """
        import tensorflow as tf

        _, embeddings, _ = self._yamnet(waveform)
        # embeddings shape: (N_frames, 1024)
        # Mean-pool -> (1024,) then add batch dim -> (1, 1024)
        pooled = tf.reduce_mean(embeddings, axis=0, keepdims=True)  # (1, 1024)
        return self._classifier(pooled, training=False)              # (1, 2)

    def __call__(self, waveform):
        return self._infer(waveform)


def export_saved_model(
    yamnet,
    classifier,
    saved_model_dir: Path,
) -> None:
    """Save the combined YAMNet + classifier as a TensorFlow SavedModel.

    Parameters
    ----------
    yamnet:
        Loaded TF Hub YAMNet handle.
    classifier:
        Trained Keras model.
    saved_model_dir:
        Directory to write the SavedModel.  Created if absent.
    """
    import tensorflow as tf

    logger.info("Saving combined SavedModel to %s", saved_model_dir)
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    combined = CryDetectorModule(yamnet, classifier)
    # Force tracing of the concrete function before saving.
    _ = combined(tf.zeros([16000], dtype=tf.float32))

    tf.saved_model.save(
        combined,
        str(saved_model_dir),
        signatures={"serving_default": combined._infer},
    )
    logger.info("SavedModel written to %s", saved_model_dir)


def export_tflite_int8(
    saved_model_dir: Path,
    tflite_output_path: Path,
    calibration_data: np.ndarray,
) -> int:
    """Convert the SavedModel to an INT8-quantised TFLite model.

    Applies full integer quantisation using a representative dataset for
    calibration.  Input and output tensors are kept as float32 for
    compatibility with the existing audio pipeline.

    Parameters
    ----------
    saved_model_dir:
        Path to the TensorFlow SavedModel directory.
    tflite_output_path:
        Path where the ``.tflite`` file will be written.  Parent directory
        must exist.
    calibration_data:
        Float32 array of shape ``(N, window_frames)`` — representative waveform
        samples used to calibrate INT8 quantisation ranges.  N should be
        100–500 for stable calibration.

    Returns
    -------
    int
        Model size in bytes.
    """
    import tensorflow as tf

    def representative_dataset():
        for i in range(min(len(calibration_data), CALIBRATION_SAMPLES)):
            waveform = calibration_data[i].astype(np.float32)
            # Yield as a list of tensors matching the model's input signature.
            yield [waveform]

    logger.info(
        "Converting to INT8 TFLite with %d calibration samples",
        min(len(calibration_data), CALIBRATION_SAMPLES),
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        # Fallback: some YAMNet ops may not have INT8 kernels; allow SELECT_TF_OPS
        # only as a last resort to keep the model size small.
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    # Keep I/O as float32 so the pipeline does not need to handle quantised I/O.
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    size_bytes = len(tflite_model)
    size_mb = size_bytes / (1024 * 1024)

    tflite_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)

    logger.info(
        "TFLite model written to %s (%.1f MB)",
        tflite_output_path,
        size_mb,
    )

    size_limit_mb = 20.0
    if size_mb > size_limit_mb:
        logger.error(
            "Model size %.1f MB exceeds spec limit of %.1f MB. "
            "Consider reducing the YAMNet base model or applying further pruning.",
            size_mb,
            size_limit_mb,
        )
    else:
        logger.info("Model size %.1f MB is within the %.1f MB spec limit", size_mb, size_limit_mb)

    return size_bytes


# ---------------------------------------------------------------------------
# FP32 baseline export (for accuracy delta measurement)
# ---------------------------------------------------------------------------


def export_tflite_fp32(
    saved_model_dir: Path,
    tflite_output_path: Path,
) -> int:
    """Convert the SavedModel to a float32 TFLite model (baseline for delta measurement).

    Parameters
    ----------
    saved_model_dir:
        Path to the TensorFlow SavedModel directory.
    tflite_output_path:
        Path where the FP32 ``.tflite`` file will be written.

    Returns
    -------
    int
        Model size in bytes.
    """
    import tensorflow as tf

    logger.info("Exporting FP32 TFLite baseline to %s", tflite_output_path)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    # No quantisation optimisations — pure FP32
    tflite_model = converter.convert()
    size_bytes = len(tflite_model)

    tflite_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)

    logger.info(
        "FP32 TFLite model written to %s (%.1f MB)",
        tflite_output_path,
        size_bytes / (1024 * 1024),
    )
    return size_bytes


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def run_training(
    repo_root: Path,
    epochs: int = 30,
    force_resample: bool = False,
    saved_model_dir: Path | None = None,
    tflite_output_path: Path | None = None,
    tflite_fp32_path: Path | None = None,
) -> int:
    """End-to-end training and export pipeline.

    Parameters
    ----------
    repo_root:
        Repository root.
    epochs:
        Maximum training epochs per attempt.
    force_resample:
        Pass to data_prep — forces re-resampling of source audio.
    saved_model_dir:
        Override output directory for the SavedModel.
    tflite_output_path:
        Override path for the INT8 TFLite output file.
    tflite_fp32_path:
        Override path for the FP32 TFLite baseline file.

    Returns
    -------
    int
        Exit code: 0 = gates passed, 1 = gates failed, 2 = fatal error.
    """
    # -- Resolve output paths -----------------------------------------------
    models_dir = repo_root / "src" / "ml" / "models"
    if saved_model_dir is None:
        saved_model_dir = repo_root / "data" / "cry_detector_saved"
    if tflite_output_path is None:
        tflite_output_path = models_dir / "cry_detection.tflite"
    if tflite_fp32_path is None:
        tflite_fp32_path = models_dir / "cry_detection_fp32.tflite"

    # -- Import TF / TF Hub -------------------------------------------------
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError as exc:
        logger.error(
            "TensorFlow or tensorflow-hub is not installed: %s\n"
            "Install with: pip install -r src/ml/training/requirements.txt",
            exc,
        )
        return 2

    logger.info("TensorFlow version: %s", tf.__version__)

    # -- Load YAMNet --------------------------------------------------------
    logger.info("Loading YAMNet from %s ...", YAMNET_HUB_URL)
    t0 = time.monotonic()
    yamnet_model = hub.load(YAMNET_HUB_URL)
    logger.info("YAMNet loaded in %.1fs", time.monotonic() - t0)

    # -- Prepare data -------------------------------------------------------
    sys.path.insert(0, str(_SCRIPT_DIR))
    from data_prep import prepare_dataset

    try:
        train_pairs, val_pairs, test_pairs = prepare_dataset(
            repo_root=repo_root,
            force_resample=force_resample,
        )
    except Exception as exc:
        logger.error("Dataset preparation failed: %s", exc)
        return 2

    # -- Extract embeddings -------------------------------------------------
    logger.info("Extracting YAMNet embeddings for training set (%d clips)...", len(train_pairs))
    X_train, y_train = extract_embeddings(train_pairs, yamnet_model)

    logger.info("Extracting YAMNet embeddings for validation set (%d clips)...", len(val_pairs))
    X_val, y_val = extract_embeddings(val_pairs, yamnet_model)

    logger.info("Extracting YAMNet embeddings for test set (%d clips)...", len(test_pairs))
    X_test, y_test = extract_embeddings(test_pairs, yamnet_model)

    # -- Training loop (with one retry on gate failure) --------------------
    precision = recall = f1 = 0.0
    dropout = DROPOUT_RATE

    for attempt in range(1, MAX_TRAINING_ATTEMPTS + 1):
        logger.info(
            "=== Training attempt %d / %d (dropout=%.2f, epochs=%d) ===",
            attempt,
            MAX_TRAINING_ATTEMPTS,
            dropout,
            epochs,
        )
        classifier = train_classifier(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            dropout_rate=dropout,
        )
        precision, recall, f1 = evaluate_classifier(classifier, X_test, y_test)

        gates_passed = precision >= PRECISION_GATE and recall >= RECALL_GATE
        if gates_passed:
            logger.info(
                "Acceptance gates PASSED: precision=%.4f (>= %.2f), recall=%.4f (>= %.2f)",
                precision, PRECISION_GATE, recall, RECALL_GATE,
            )
            break

        if attempt < MAX_TRAINING_ATTEMPTS:
            logger.warning(
                "Gates not met on attempt %d "
                "(precision=%.4f, recall=%.4f). "
                "Retrying with dropout=%.2f and %d additional epochs.",
                attempt, precision, recall, DROPOUT_RATE_RETRY, epochs,
            )
            dropout = DROPOUT_RATE_RETRY
        else:
            logger.error(
                "Acceptance gates FAILED after %d attempts:\n"
                "  Precision = %.4f  (gate >= %.2f)  %s\n"
                "  Recall    = %.4f  (gate >= %.2f)  %s\n"
                "Escalate to product-manager per TASK-201 escalation path.",
                MAX_TRAINING_ATTEMPTS,
                precision, PRECISION_GATE, "PASS" if precision >= PRECISION_GATE else "FAIL",
                recall, RECALL_GATE, "PASS" if recall >= RECALL_GATE else "FAIL",
            )

    # -- Export SavedModel --------------------------------------------------
    logger.info("Exporting combined SavedModel...")
    try:
        export_saved_model(yamnet_model, classifier, saved_model_dir)
    except Exception as exc:
        logger.error("SavedModel export failed: %s", exc)
        return 2

    # -- Build calibration dataset for INT8 conversion ---------------------
    # Use a random sample of training waveforms (raw audio, not embeddings).
    logger.info("Loading calibration waveforms for INT8 quantisation...")
    from data_prep import load_wav_16k

    rng = np.random.default_rng(seed=0)
    calib_indices = rng.choice(
        len(train_pairs),
        size=min(CALIBRATION_SAMPLES, len(train_pairs)),
        replace=False,
    )
    calib_waveforms: List[np.ndarray] = []
    for idx in calib_indices:
        wav_path, _ = train_pairs[int(idx)]
        try:
            audio = load_wav_16k(wav_path)
            calib_waveforms.append(audio)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Calibration load failed for %s: %s", wav_path, exc)

    calib_array = np.array(calib_waveforms, dtype=object)  # jagged — kept as list

    # -- Export FP32 baseline -----------------------------------------------
    logger.info("Exporting FP32 TFLite baseline...")
    try:
        export_tflite_fp32(saved_model_dir, tflite_fp32_path)
    except Exception as exc:
        logger.warning("FP32 TFLite export failed (non-fatal): %s", exc)

    # -- Export INT8 TFLite -------------------------------------------------
    logger.info("Exporting INT8 TFLite model...")
    try:
        export_tflite_int8(saved_model_dir, tflite_output_path, calib_waveforms)
    except Exception as exc:
        logger.error("INT8 TFLite export failed: %s", exc)
        return 2

    # -- Print final summary ------------------------------------------------
    _print_summary(
        tflite_output_path=tflite_output_path,
        precision=precision,
        recall=recall,
        f1=f1,
        precision_gate=PRECISION_GATE,
        recall_gate=RECALL_GATE,
    )

    gates_passed = precision >= PRECISION_GATE and recall >= RECALL_GATE
    return 0 if gates_passed else 1


def _print_summary(
    tflite_output_path: Path,
    precision: float,
    recall: float,
    f1: float,
    precision_gate: float,
    recall_gate: float,
) -> None:
    """Print a human-readable training summary."""
    size_str = "N/A"
    if tflite_output_path.exists():
        size_mb = tflite_output_path.stat().st_size / (1024 * 1024)
        size_str = f"{size_mb:.2f} MB"

    precision_status = "PASS" if precision >= precision_gate else "FAIL"
    recall_status = "PASS" if recall >= recall_gate else "FAIL"

    print("\n" + "=" * 65)
    print("YAMNet Cry Detector — Training Summary")
    print("=" * 65)
    print(f"  Output model:  {tflite_output_path}")
    print(f"  Model size:    {size_str}  (spec limit: <= 20 MB)")
    print()
    print(f"  Precision:     {precision:.4f}  (gate >= {precision_gate:.2f}) [{precision_status}]")
    print(f"  Recall:        {recall:.4f}  (gate >= {recall_gate:.2f}) [{recall_status}]")
    print(f"  F1:            {f1:.4f}")
    print()
    all_pass = precision >= precision_gate and recall >= recall_gate
    print(f"  Acceptance gates: {'ALL PASSED' if all_pass else 'FAILED'}")
    print("=" * 65)
    print()
    if all_pass:
        print("Next step: benchmark on Raspberry Pi 4:")
        print(f"  scp {tflite_output_path} pi@<PI_IP>:~/baby_monitor/src/ml/models/")
        print("  python src/ml/utils/benchmark_model.py --model src/ml/models/cry_detection.tflite --n-runs 200")
        print()
        print("Then version and symlink per TASK-201 Step 10:")
        print("  mv src/ml/models/cry_detection.tflite src/ml/models/cry_detection_v1_int8.tflite")
        print("  ln -sf cry_detection_v1_int8.tflite src/ml/models/cry_detection.tflite")
    else:
        print("Gates not met. See TASK-201 escalation path in docs/tasks/TASK-201-tflite-model-setup.md")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """Command-line entry point.

    Returns
    -------
    int
        Exit code: 0 = success + gates passed, 1 = gates failed, 2 = fatal error.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Train YAMNet-based cry detector and export INT8 TFLite model. "
            "Run on a dev machine — do NOT run on Raspberry Pi."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        default=str(_DEFAULT_REPO_ROOT),
        help="Repository root directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum training epochs per attempt",
    )
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="Force re-resampling of source audio even if processed files exist",
    )
    parser.add_argument(
        "--saved-model-dir",
        default=None,
        help="Override SavedModel output directory (default: data/cry_detector_saved/)",
    )
    parser.add_argument(
        "--tflite-output",
        default=None,
        help="Override TFLite INT8 output path (default: src/ml/models/cry_detection.tflite)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    saved_model_dir = Path(args.saved_model_dir).resolve() if args.saved_model_dir else None
    tflite_output = Path(args.tflite_output).resolve() if args.tflite_output else None

    logger.info("Repository root: %s", repo_root)
    logger.info("Epochs: %d", args.epochs)

    return run_training(
        repo_root=repo_root,
        epochs=args.epochs,
        force_resample=args.force_resample,
        saved_model_dir=saved_model_dir,
        tflite_output_path=tflite_output,
    )


if __name__ == "__main__":
    sys.exit(main())
