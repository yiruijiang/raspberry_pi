"""
cry_detection_pipeline.py — End-to-end cry detection process entry point.

This module wires together AudioCapture, AudioPreprocessor, CryDetector,
and SocketPublisher into a single long-running inference loop suitable for
running as a systemd service on the Raspberry Pi.

Process lifecycle
-----------------
1. Parse config from environment variables.
2. Initialise all components and validate model shape.
3. Start audio capture thread and socket publisher.
4. Run sliding-window inference loop indefinitely.
5. On SIGTERM / SIGINT, shut down gracefully (flush socket, stop capture).

CPU budget
----------
The spec requires <= 30% CPU on a single core.  This is achieved by:
- Using tflite-runtime (not full TensorFlow) with XNNPACK delegate.
- Running at 1-second windows with 50% overlap → 2 inferences per second.
- Sleeping for ``hop_seconds`` between inference calls so the loop does not
  busy-wait.

Latency
-------
End-to-end path: microphone → PyAudio buffer → feature extraction → TFLite
inference → SocketPublisher.publish().  Target: <= 200 ms.

Typical RPi 4 breakdown (estimated, benchmark on hardware):
    - Audio capture buffering:        ~0 ms (background thread)
    - MFCC extraction (librosa):      ~10 ms
    - TFLite INT8 inference:          ~20–50 ms
    - JSON serialise + socket send:   ~1 ms
    Total:                            ~30–60 ms  (well within 200 ms budget)

Usage (direct)
--------------
    python -m pipelines.cry_detection_pipeline

Usage (as systemd service)
--------------------------
    ExecStart=/usr/bin/python3 -m pipelines.cry_detection_pipeline
    Environment=BABY_MONITOR_MODEL_PATH=/opt/baby_monitor/models/cry_detection.tflite
    Environment=BABY_MONITOR_SOCKET_PATH=/run/baby_monitor/ml.sock
    Environment=BABY_MONITOR_CRY_THRESHOLD=0.75
"""

import logging
import signal
import sys
import time

# Ensure src/ml/ is on the path when invoked as a module.
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_capture import AudioCapture
from config import Config
from cry_detector import CryDetector
from socket_publisher import SocketPublisher

logger = logging.getLogger(__name__)


class CryDetectionPipeline:
    """Orchestrates the full cry detection inference loop.

    Parameters
    ----------
    config:
        Top-level :class:`~config.Config` instance.  Pass a custom instance
        to override defaults in tests.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._capture = AudioCapture(config.audio)
        self._detector = CryDetector(config.audio, config.model)
        self._publisher = SocketPublisher(config.socket)
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the pipeline components (non-blocking; call run() to loop)."""
        logger.info("Starting cry detection pipeline")
        self._publisher.start()
        self._capture.start()
        self._running = True
        logger.info("Pipeline components started")

    def run(self) -> None:
        """Run the inference loop until stop() is called or a fatal error occurs.

        This method blocks.  Call it from the main thread after start().
        """
        cfg = self._cfg.audio
        hop_seconds = cfg.window_seconds * (1.0 - cfg.window_overlap)

        logger.info(
            "Entering inference loop — window=%.2fs overlap=%.0f%% hop=%.2fs",
            cfg.window_seconds,
            cfg.window_overlap * 100,
            hop_seconds,
        )

        # Give the capture thread time to fill its first window.
        warmup = cfg.window_seconds
        logger.debug("Warming up audio buffer for %.2fs", warmup)
        time.sleep(warmup)

        while self._running:
            loop_start = time.monotonic()

            window = self._capture.read_window()
            if window is None:
                logger.debug("Waiting for audio buffer to fill")
                time.sleep(0.05)
                continue

            try:
                event = self._detector.infer(window)
            except Exception as exc:
                logger.exception("Inference error (skipping window): %s", exc)
                time.sleep(hop_seconds)
                continue

            if event.event_type == "cry":
                sent = self._publisher.publish(event.to_dict())
                logger.info(
                    "CRY detected — confidence=%.3f sent_to_socket=%s",
                    event.confidence,
                    sent,
                )
            else:
                logger.debug(
                    "Silence — confidence=%.3f",
                    event.confidence,
                )

            # Sleep for the remainder of the hop period so we don't busy-spin.
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, hop_seconds - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self) -> None:
        """Shut down the pipeline gracefully."""
        logger.info("Stopping cry detection pipeline")
        self._running = False
        self._capture.stop()
        self._publisher.stop()
        logger.info("Pipeline stopped")


def _setup_signal_handlers(pipeline: CryDetectionPipeline) -> None:
    """Register SIGTERM and SIGINT handlers for clean shutdown."""

    def _handler(signum, _frame):
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — initiating shutdown", sig_name)
        pipeline.stop()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def main() -> int:
    """Entry point for the cry detection pipeline process.

    Returns
    -------
    int
        Exit code: 0 on clean shutdown, 1 on startup failure.
    """
    cfg = Config()
    cfg.configure_logging()

    logger.info("Baby Monitor — Cry Detection Pipeline starting (PID=%d)", os.getpid())
    logger.info(
        "Config: model=%s socket=%s threshold=%.2f sample_rate=%d",
        cfg.model.model_path,
        cfg.socket.socket_path,
        cfg.model.cry_threshold,
        cfg.audio.sample_rate,
    )

    pipeline = CryDetectionPipeline(cfg)
    _setup_signal_handlers(pipeline)

    try:
        pipeline.start()
    except FileNotFoundError as exc:
        logger.error("Model file not found: %s", exc)
        logger.error("See src/ml/models/README.md for setup instructions")
        return 1
    except RuntimeError as exc:
        logger.error("Pipeline startup failed: %s", exc)
        return 1

    pipeline.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
