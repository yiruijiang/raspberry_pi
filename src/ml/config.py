"""
config.py — Centralised configuration for the ML audio/video pipelines.

All runtime-tunable values are read from environment variables with safe
defaults.  No paths or thresholds are hard-coded anywhere else in src/ml/.

Environment variables (all optional — defaults shown):
    BABY_MONITOR_MODEL_PATH         Path to the TFLite cry-detection model.
                                    Default: src/ml/models/cry_detection.tflite
    BABY_MONITOR_SOCKET_PATH        Unix domain socket path written by the
                                    socket publisher and read by the backend.
                                    Default: /tmp/baby_monitor_ml.sock
    BABY_MONITOR_SAMPLE_RATE        Microphone sample rate in Hz. Default: 16000
    BABY_MONITOR_CHANNELS           Number of audio channels.  Default: 1
    BABY_MONITOR_CHUNK_FRAMES       PyAudio frames per read chunk. Default: 1024
    BABY_MONITOR_WINDOW_SECONDS     Sliding inference window length (s). Default: 1.0
    BABY_MONITOR_WINDOW_OVERLAP     Fraction of window to overlap [0, 1). Default: 0.5
    BABY_MONITOR_CRY_THRESHOLD      Minimum confidence to emit a cry event [0,1].
                                    Default: 0.75
    BABY_MONITOR_N_MFCC             Number of MFCC coefficients. Default: 40
    BABY_MONITOR_N_FFT              FFT window size for feature extraction. Default: 512
    BABY_MONITOR_HOP_LENGTH         Hop length for STFT in feature extraction. Default: 160
    BABY_MONITOR_N_MELS             Number of mel filterbank bins. Default: 64
    BABY_MONITOR_LOG_LEVEL          Python logging level name. Default: INFO
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key}={val!r} is not a valid float") from exc


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key}={val!r} is not a valid integer") from exc


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Default filesystem locations (relative to repository root when running on Pi)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_PATH = str(_REPO_ROOT / "src" / "ml" / "models" / "cry_detection.tflite")
_DEFAULT_SOCKET_PATH = "/tmp/baby_monitor_ml.sock"


@dataclass(frozen=True)
class AudioConfig:
    """Parameters controlling microphone capture and feature extraction."""

    # Capture
    sample_rate: int = field(default_factory=lambda: _env_int("BABY_MONITOR_SAMPLE_RATE", 16_000))
    channels: int = field(default_factory=lambda: _env_int("BABY_MONITOR_CHANNELS", 1))
    chunk_frames: int = field(default_factory=lambda: _env_int("BABY_MONITOR_CHUNK_FRAMES", 1024))

    # Sliding-window inference
    window_seconds: float = field(
        default_factory=lambda: _env_float("BABY_MONITOR_WINDOW_SECONDS", 1.0)
    )
    window_overlap: float = field(
        default_factory=lambda: _env_float("BABY_MONITOR_WINDOW_OVERLAP", 0.5)
    )

    # MFCC / mel-spectrogram
    n_mfcc: int = field(default_factory=lambda: _env_int("BABY_MONITOR_N_MFCC", 40))
    n_fft: int = field(default_factory=lambda: _env_int("BABY_MONITOR_N_FFT", 512))
    hop_length: int = field(default_factory=lambda: _env_int("BABY_MONITOR_HOP_LENGTH", 160))
    n_mels: int = field(default_factory=lambda: _env_int("BABY_MONITOR_N_MELS", 64))

    def __post_init__(self) -> None:
        if not (0.0 <= self.window_overlap < 1.0):
            raise ValueError(
                f"window_overlap must be in [0, 1); got {self.window_overlap}"
            )
        if self.window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive; got {self.window_seconds}")

    @property
    def window_frames(self) -> int:
        """Number of PCM samples in one inference window."""
        return int(self.sample_rate * self.window_seconds)

    @property
    def hop_frames(self) -> int:
        """Number of PCM samples to advance between consecutive windows."""
        return int(self.window_frames * (1.0 - self.window_overlap))


@dataclass(frozen=True)
class ModelConfig:
    """Parameters for the TFLite model and inference."""

    model_path: str = field(
        default_factory=lambda: _env_str("BABY_MONITOR_MODEL_PATH", _DEFAULT_MODEL_PATH)
    )
    cry_threshold: float = field(
        default_factory=lambda: _env_float("BABY_MONITOR_CRY_THRESHOLD", 0.75)
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.cry_threshold <= 1.0):
            raise ValueError(
                f"cry_threshold must be in [0, 1]; got {self.cry_threshold}"
            )


@dataclass(frozen=True)
class SocketConfig:
    """Parameters for the Unix domain socket publisher."""

    socket_path: str = field(
        default_factory=lambda: _env_str("BABY_MONITOR_SOCKET_PATH", _DEFAULT_SOCKET_PATH)
    )


@dataclass(frozen=True)
class Config:
    """Top-level configuration container.  Instantiate once per process."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    socket: SocketConfig = field(default_factory=SocketConfig)
    log_level: str = field(
        default_factory=lambda: _env_str("BABY_MONITOR_LOG_LEVEL", "INFO")
    )

    def configure_logging(self) -> None:
        """Apply log_level to the root logger."""
        numeric = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric,
            format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )


# Module-level singleton — import and use directly where convenient.
config = Config()
