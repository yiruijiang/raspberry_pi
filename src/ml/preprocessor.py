"""
preprocessor.py — Audio feature extraction for cry detection inference.

Feature pipeline
----------------
Input: raw float32 PCM window, shape ``(window_frames,)``, values in ``[-1, 1]``.

Output: 2-D feature map ready for TFLite input, shape ``(1, n_mfcc, time_steps, 1)``
        where the leading ``1`` is the batch dimension and the trailing ``1`` is the
        channel dimension (expected by a Conv2D-based classifier).

Features computed
-----------------
- **Log-mel spectrogram** (``n_mels`` bands): captures the frequency envelope that
  distinguishes infant cry (fundamental ~250–600 Hz, prominent harmonics) from
  ambient noise.
- **MFCCs** (``n_mfcc`` coefficients): compact representation derived from the
  log-mel spectrogram.  MFCCs are the standard representation for lightweight
  audio classifiers on constrained hardware.

We use librosa for feature extraction.  Its numpy-based implementation avoids
any GPU dependency and is well-suited to ARM Cortex-A72 (RPi 4).

Latency budget
--------------
Typical librosa MFCC extraction for a 1-second window at 16 kHz:
    ~5–15 ms on RPi 4 (single core, no JIT warm-up overhead).
This is well within the 200 ms end-to-end budget.

Usage
-----
    from preprocessor import AudioPreprocessor
    from config import config

    pre = AudioPreprocessor(config.audio)
    features = pre.extract(pcm_window)   # shape (1, 40, T, 1)
"""

import logging
from typing import Tuple

import librosa
import numpy as np

from config import AudioConfig

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Converts raw PCM windows into normalised MFCC feature tensors.

    Parameters
    ----------
    config:
        :class:`~config.AudioConfig` instance.  Reads ``sample_rate``,
        ``n_mfcc``, ``n_fft``, ``hop_length``, and ``n_mels``.

    Notes
    -----
    A per-instance running statistics object is maintained so that future
    versions can implement online normalisation (mean/variance computed over
    a rolling history of windows instead of per-window).  Currently,
    per-window standardisation (zero mean, unit variance) is used because
    it requires no warmup period and is robust to microphone gain drift.
    """

    def __init__(self, config: AudioConfig) -> None:
        self._cfg = config
        logger.debug(
            "AudioPreprocessor initialised — n_mfcc=%d, n_fft=%d, hop=%d, n_mels=%d",
            config.n_mfcc,
            config.n_fft,
            config.hop_length,
            config.n_mels,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, pcm_window: np.ndarray) -> np.ndarray:
        """Extract MFCCs from a single PCM window.

        Parameters
        ----------
        pcm_window:
            1-D float32 array of shape ``(window_frames,)`` with values in
            ``[-1, 1]``.  Must have at least ``n_fft`` samples.

        Returns
        -------
        numpy.ndarray
            Float32 tensor of shape ``(1, n_mfcc, time_steps, 1)`` ready for
            direct assignment to a TFLite input tensor.  Values are
            per-window standardised (zero mean, unit std).

        Raises
        ------
        ValueError
            If ``pcm_window`` is too short for the configured ``n_fft``.
        """
        if pcm_window.ndim != 1:
            raise ValueError(
                f"pcm_window must be 1-D, got shape {pcm_window.shape}"
            )
        if len(pcm_window) < self._cfg.n_fft:
            raise ValueError(
                f"pcm_window length {len(pcm_window)} is less than n_fft {self._cfg.n_fft}"
            )

        # Ensure float32 and clip to [-1, 1] to guard against clipping artefacts.
        audio = np.clip(pcm_window.astype(np.float32), -1.0, 1.0)

        mfccs = self._compute_mfccs(audio)          # (n_mfcc, time_steps)
        mfccs = self._standardise(mfccs)             # zero-mean / unit-std
        tensor = mfccs[np.newaxis, :, :, np.newaxis]  # (1, n_mfcc, time_steps, 1)
        return tensor.astype(np.float32)

    def feature_shape(self) -> Tuple[int, int]:
        """Return ``(n_mfcc, expected_time_steps)`` for the configured window.

        Useful for verifying that the TFLite model input shape matches the
        preprocessor output before starting the pipeline.
        """
        dummy_frames = self._cfg.window_frames
        dummy_audio = np.zeros(dummy_frames, dtype=np.float32)
        mfccs = self._compute_mfccs(dummy_audio)
        return mfccs.shape  # (n_mfcc, time_steps)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_mfccs(self, audio: np.ndarray) -> np.ndarray:
        """Compute MFCC matrix from a mono float32 audio array.

        Parameters
        ----------
        audio:
            1-D float32 array, values in ``[-1, 1]``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_mfcc, time_steps)``, dtype float32.
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self._cfg.sample_rate,
            n_mfcc=self._cfg.n_mfcc,
            n_fft=self._cfg.n_fft,
            hop_length=self._cfg.hop_length,
            n_mels=self._cfg.n_mels,
            fmin=50.0,    # minimum frequency: below fundamental infant cry (~250 Hz) to
                          # capture sub-harmonics; well above DC offset
            fmax=8000.0,  # maximum frequency: 8 kHz captures all diagnostically relevant
                          # harmonics while ignoring high-freq sensor noise on budget mics
        )
        return mfccs.astype(np.float32)

    @staticmethod
    def _standardise(mfccs: np.ndarray) -> np.ndarray:
        """Normalise MFCC matrix to zero mean and unit standard deviation.

        Standardisation is computed across the entire feature map (all
        coefficients and all time frames jointly) so that the scale is
        consistent regardless of window length or n_mfcc setting.

        Parameters
        ----------
        mfccs:
            Shape ``(n_mfcc, time_steps)``.

        Returns
        -------
        numpy.ndarray
            Same shape as input, dtype float32.
        """
        mean = mfccs.mean()
        std = mfccs.std()
        if std < 1e-6:
            # Silent or near-silent window — return zeros rather than NaN.
            return np.zeros_like(mfccs)
        return ((mfccs - mean) / std).astype(np.float32)
