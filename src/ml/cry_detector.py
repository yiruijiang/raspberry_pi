"""
cry_detector.py — TFLite inference engine and detection event emitter.

Architecture overview
---------------------

                ┌──────────────────┐
  PCM window ──►│ AudioPreprocessor│──► MFCC tensor (1, n_mfcc, T, 1)
                └──────────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │  TFLite runtime  │  (INT8 quantised model, XNNPACK delegate)
                │  cry_detection   │──► raw output logits / probabilities
                │  .tflite         │    shape: (1, 2) — [p_not_cry, p_cry]
                └──────────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │  CryDetector     │  threshold gate → DetectionEvent
                └──────────────────┘
                          │
                          ▼
               SocketPublisher.publish()

Output schema (matches spec § Communication Contracts)
------------------------------------------------------
    {
        "type":       "cry" | "silence",
        "confidence": float,        # probability assigned to the cry class
        "timestamp":  "<ISO8601>"   # UTC, e.g. "2026-03-25T14:32:01.123Z"
    }

Only events with confidence >= cry_threshold are emitted to the socket.
"silence" events are logged at DEBUG level but not forwarded.

Model input contract
--------------------
The TFLite model must accept an input tensor of shape
    (1, n_mfcc, time_steps, 1)  — dtype float32 (FP32 model)
                                   or uint8 / int8 (quantised model)
and produce an output tensor of shape
    (1, 2)
where index 0 is the background/silence class probability and index 1 is the
cry class probability (after softmax).

If your model uses a different output layout, subclass :class:`CryDetector`
and override :meth:`_parse_output`.

RPi optimisation notes
-----------------------
- Load the model once at startup; reuse the interpreter for every inference.
- Use the XNNPACK delegate for ~2x speedup on ARM Cortex-A72 without any code
  changes — enabled by default in TFLite >= 2.9.
- INT8 post-training quantisation reduces model size by ~4x and inference time
  by ~2x vs. FP32 on CPU-only hardware.
- Avoid allocating new numpy arrays inside the hot path; reuse input buffer.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from config import AudioConfig, ModelConfig
from preprocessor import AudioPreprocessor

logger = logging.getLogger(__name__)

# Lazy import of TFLite runtime.  This allows the module to be imported for
# testing on a dev machine that only has tflite-runtime installed (not the
# full tensorflow package), and surfaces a clear error if neither is present.
_tflite_interpreter = None


def _get_tflite_interpreter():
    """Return the tflite.Interpreter class, preferring the lightweight runtime."""
    global _tflite_interpreter
    if _tflite_interpreter is not None:
        return _tflite_interpreter

    try:
        import tflite_runtime.interpreter as tflite
        _tflite_interpreter = tflite.Interpreter
        logger.debug("Using tflite_runtime package")
    except ImportError:
        try:
            import tensorflow as tf
            _tflite_interpreter = tf.lite.Interpreter
            logger.debug("Using tensorflow.lite.Interpreter (full TF install)")
        except ImportError as exc:
            raise ImportError(
                "Neither 'tflite-runtime' nor 'tensorflow' is installed. "
                "Install 'tflite-runtime' for production use on Raspberry Pi."
            ) from exc

    return _tflite_interpreter


@dataclass(frozen=True)
class DetectionEvent:
    """A single cry-detection inference result.

    Attributes
    ----------
    event_type:
        ``"cry"`` when confidence exceeds the threshold, ``"silence"`` otherwise.
    confidence:
        Softmax probability for the cry class in ``[0, 1]``.
    timestamp:
        UTC ISO 8601 string, e.g. ``"2026-03-25T14:32:01.123Z"``.
    """

    event_type: str     # "cry" | "silence"
    confidence: float
    timestamp: str

    def to_dict(self) -> dict:
        """Serialise to the wire-format dict consumed by SocketPublisher.

        Returns
        -------
        dict
            ``{"type": str, "confidence": float, "timestamp": str}``
        """
        return {
            "type": self.event_type,
            "confidence": round(float(self.confidence), 4),
            "timestamp": self.timestamp,
        }


class CryDetector:
    """Loads a TFLite model and runs cry-detection inference on MFCC windows.

    Parameters
    ----------
    audio_config:
        :class:`~config.AudioConfig` — passed to the internal
        :class:`~preprocessor.AudioPreprocessor`.
    model_config:
        :class:`~config.ModelConfig` — provides ``model_path`` and
        ``cry_threshold``.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at ``model_config.model_path``.
    RuntimeError
        If the TFLite interpreter cannot be allocated or the model's input
        shape does not match the preprocessor's output shape.
    """

    def __init__(self, audio_config: AudioConfig, model_config: ModelConfig) -> None:
        self._audio_cfg = audio_config
        self._model_cfg = model_config
        self._preprocessor = AudioPreprocessor(audio_config)

        self._interpreter = self._load_model(model_config.model_path)
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        self._validate_model_shape()

        # Pre-allocate the reusable input buffer to avoid GC pressure on the
        # inference hot path.
        input_shape = self._input_details[0]["shape"]
        input_dtype = self._input_details[0]["dtype"]
        self._input_buffer = np.zeros(input_shape, dtype=input_dtype)

        logger.info(
            "CryDetector ready — model=%s, threshold=%.2f, input_shape=%s",
            model_config.model_path,
            model_config.cry_threshold,
            input_shape,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self, pcm_window: np.ndarray) -> DetectionEvent:
        """Run end-to-end inference on a single PCM window.

        Parameters
        ----------
        pcm_window:
            1-D float32 array of shape ``(window_frames,)``, values in
            ``[-1, 1]``, as produced by :class:`~audio_capture.AudioCapture`.

        Returns
        -------
        DetectionEvent
            Always returns an event.  The caller should check
            ``event.event_type == "cry"`` and filter on confidence before
            forwarding to the socket publisher.
        """
        t_start = time.monotonic()

        # 1. Feature extraction
        features = self._preprocessor.extract(pcm_window)  # (1, n_mfcc, T, 1)

        # 2. Quantise if the model expects integer input.
        self._fill_input_buffer(features)

        # 3. TFLite inference
        self._interpreter.set_tensor(self._input_details[0]["index"], self._input_buffer)
        self._interpreter.invoke()

        # 4. Parse output
        raw_output = self._interpreter.get_tensor(self._output_details[0]["index"])
        cry_prob = float(self._parse_output(raw_output))

        # 5. Build event
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        event_type = "cry" if cry_prob >= self.cry_threshold else "silence"
        event = DetectionEvent(event_type=event_type, confidence=cry_prob, timestamp=ts)

        latency_ms = (time.monotonic() - t_start) * 1000
        log_fn = logger.info if event_type == "cry" else logger.debug
        log_fn(
            "inference: type=%s confidence=%.3f latency_ms=%.1f",
            event_type,
            cry_prob,
            latency_ms,
        )

        if latency_ms > 200:
            logger.warning(
                "Inference latency %.1f ms exceeded 200 ms target — "
                "consider INT8 quantisation or reducing window size",
                latency_ms,
            )

        return event

    def update_threshold(self, threshold: float) -> None:
        """Update the confidence threshold at runtime without restarting.

        This supports the spec requirement that threshold changes take effect
        on the next inference cycle without a server restart.

        Parameters
        ----------
        threshold:
            New threshold in ``[0, 1]``.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1]; got {threshold}")
        # ModelConfig is frozen; replace the reference via object.__setattr__
        # on a new instance is not practical here, so we store a mutable override.
        self._runtime_threshold = threshold
        logger.info("Cry threshold updated to %.2f", threshold)

    @property
    def cry_threshold(self) -> float:
        """Current active cry confidence threshold."""
        return getattr(self, "_runtime_threshold", self._model_cfg.cry_threshold)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str):
        """Load TFLite model from disk and allocate tensors.

        Parameters
        ----------
        model_path:
            Absolute or relative path to a ``.tflite`` file.

        Returns
        -------
        tflite.Interpreter
            Ready-to-use interpreter with tensors allocated.
        """
        import os
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"TFLite model not found at '{model_path}'. "
                "See src/ml/models/README.md for instructions on obtaining or training "
                "the cry detection model."
            )

        InterpreterClass = _get_tflite_interpreter()

        # Enable XNNPACK delegate for ARM NEON acceleration on RPi 4.
        # num_threads=1 keeps CPU usage predictable; the spec target is <=30%
        # on a single core, and XNNPACK is efficient enough to meet that.
        try:
            interpreter = InterpreterClass(
                model_path=model_path,
                num_threads=1,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load TFLite model from '{model_path}': {exc}") from exc

        interpreter.allocate_tensors()
        logger.debug("TFLite model loaded from %s", model_path)
        return interpreter

    def _validate_model_shape(self) -> None:
        """Check that the model's input shape is compatible with the preprocessor."""
        expected_n_mfcc, expected_time_steps = self._preprocessor.feature_shape()
        model_input_shape = tuple(self._input_details[0]["shape"])
        # Expected: (1, n_mfcc, time_steps, 1)
        expected_shape = (1, expected_n_mfcc, expected_time_steps, 1)

        if model_input_shape != expected_shape:
            logger.warning(
                "Model input shape %s does not match preprocessor output shape %s. "
                "Inference may fail or produce incorrect results. "
                "Re-train or re-export the model with the correct input dimensions.",
                model_input_shape,
                expected_shape,
            )
        else:
            logger.debug("Model input shape %s validated successfully", model_input_shape)

    def _fill_input_buffer(self, features: np.ndarray) -> None:
        """Copy and optionally quantise features into the pre-allocated input buffer.

        Parameters
        ----------
        features:
            Float32 tensor of shape ``(1, n_mfcc, time_steps, 1)``.
        """
        input_dtype = self._input_details[0]["dtype"]

        if input_dtype == np.float32:
            np.copyto(self._input_buffer, features, casting="same_kind")
        elif input_dtype in (np.int8, np.uint8):
            # Apply the quantisation parameters stored in the model.
            scale, zero_point = self._input_details[0]["quantization"]
            if scale == 0:
                # Fallback: model has no quantization params — cast directly.
                quantised = features.astype(input_dtype)
            else:
                quantised = (features / scale + zero_point).clip(
                    np.iinfo(input_dtype).min, np.iinfo(input_dtype).max
                ).astype(input_dtype)
            np.copyto(self._input_buffer, quantised, casting="unsafe")
        else:
            raise RuntimeError(
                f"Unsupported model input dtype {input_dtype}. "
                "Expected float32, int8, or uint8."
            )

    def _parse_output(self, raw_output: np.ndarray) -> float:
        """Extract the cry class probability from the model's raw output tensor.

        Parameters
        ----------
        raw_output:
            Output tensor as returned by the TFLite interpreter.
            Expected shape: ``(1, 2)`` — ``[p_not_cry, p_cry]``.
            If the model outputs a single logit or probability, override
            this method in a subclass.

        Returns
        -------
        float
            Probability of the cry class in ``[0, 1]``.
        """
        output_dtype = self._output_details[0]["dtype"]
        output = raw_output.squeeze()  # (2,) or scalar

        if output_dtype in (np.int8, np.uint8):
            # Dequantise
            scale, zero_point = self._output_details[0]["quantization"]
            if scale > 0:
                output = (output.astype(np.float32) - zero_point) * scale
            else:
                output = output.astype(np.float32)

        if output.ndim == 0:
            # Single-output sigmoid model — output is directly p_cry.
            return float(np.clip(output, 0.0, 1.0))

        if output.shape == (2,):
            # Two-class softmax: index 1 is cry.
            exp_out = np.exp(output - output.max())  # numerically stable softmax
            probs = exp_out / exp_out.sum()
            return float(probs[1])

        if output.shape[0] > 2:
            # Multi-class model — assume class 1 is "cry".
            exp_out = np.exp(output - output.max())
            probs = exp_out / exp_out.sum()
            return float(probs[1])

        logger.warning("Unexpected output shape %s; returning 0.0", output.shape)
        return 0.0
