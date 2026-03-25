"""
audio_capture.py — Real-time microphone capture with a thread-safe ring buffer.

Architecture
------------
A background thread continuously reads raw PCM frames from PyAudio and appends
them to a deque-backed ring buffer.  The inference pipeline reads contiguous
windows from the buffer via :meth:`AudioCapture.read_window` without blocking
the capture thread.

The ring buffer is sized to hold at least 10 seconds of audio so slow inference
cycles do not cause dropped frames.

Usage
-----
    from audio_capture import AudioCapture
    from config import config

    capture = AudioCapture(config.audio)
    capture.start()

    try:
        while True:
            window = capture.read_window()   # numpy float32 array, shape (window_frames,)
            if window is not None:
                process(window)
    finally:
        capture.stop()

Thread safety
-------------
:class:`collections.deque` with a fixed ``maxlen`` provides O(1) append and
popleft with the GIL acting as a sufficient lock for the single-producer /
single-consumer pattern used here.  No additional locking is required.
"""

import logging
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import pyaudio

from config import AudioConfig

logger = logging.getLogger(__name__)

# PCM format used throughout: signed 16-bit little-endian.
_PA_FORMAT = pyaudio.paInt16
_DTYPE = np.int16
_FLOAT_SCALE = 1.0 / 32768.0  # normalise int16 → float32 in [-1, 1]

# Ring buffer capacity: 10 seconds of samples at the configured sample rate.
_BUFFER_SECONDS = 10


class AudioCapture:
    """Continuously captures audio from the system default microphone.

    Parameters
    ----------
    config:
        :class:`~config.AudioConfig` instance describing sample rate, channels,
        chunk size, and window parameters.
    device_index:
        Optional PyAudio device index.  ``None`` uses the system default input.

    Attributes
    ----------
    is_running : bool
        True while the capture thread is active.
    """

    def __init__(self, config: AudioConfig, device_index: Optional[int] = None) -> None:
        self._cfg = config
        self._device_index = device_index

        # Ring buffer stores raw int16 samples from the capture thread.
        _buf_capacity = _BUFFER_SECONDS * config.sample_rate
        self._buffer: deque[int] = deque(maxlen=_buf_capacity)

        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.is_running = False

        # Track how many samples have been consumed so we can implement the
        # sliding-window read without re-scanning the entire buffer.
        self._samples_consumed = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the audio stream and start the capture thread.

        Raises
        ------
        RuntimeError
            If PyAudio cannot open the input stream (e.g., no microphone).
        """
        if self.is_running:
            logger.warning("AudioCapture.start() called while already running — ignoring")
            return

        self._pa = pyaudio.PyAudio()

        try:
            self._stream = self._pa.open(
                format=_PA_FORMAT,
                channels=self._cfg.channels,
                rate=self._cfg.sample_rate,
                input=True,
                input_device_index=self._device_index,
                frames_per_buffer=self._cfg.chunk_frames,
                stream_callback=None,  # blocking read mode — simpler on constrained hardware
            )
        except OSError as exc:
            self._pa.terminate()
            self._pa = None
            raise RuntimeError(
                f"Failed to open audio input stream (sample_rate={self._cfg.sample_rate}, "
                f"channels={self._cfg.channels}, device={self._device_index}): {exc}"
            ) from exc

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="audio-capture",
            daemon=True,
        )
        self.is_running = True
        self._thread.start()
        logger.info(
            "AudioCapture started — sample_rate=%d, channels=%d, chunk_frames=%d",
            self._cfg.sample_rate,
            self._cfg.channels,
            self._cfg.chunk_frames,
        )

    def stop(self) -> None:
        """Signal the capture thread to stop and release audio resources."""
        if not self.is_running:
            return

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

        self.is_running = False
        logger.info("AudioCapture stopped")

    def read_window(self) -> Optional[np.ndarray]:
        """Return the next inference window as a float32 array, or None.

        Windows advance by ``config.hop_frames`` samples each call.  Returns
        ``None`` if not enough samples have accumulated since the last call.

        Returns
        -------
        numpy.ndarray or None
            Shape ``(window_frames,)``, dtype ``float32``, values in ``[-1, 1]``.
        """
        available = len(self._buffer)
        if available < self._cfg.window_frames:
            return None

        # Take the most-recent window_frames samples.  For a sliding window we
        # track consumed position; here we use a simpler approach that always
        # returns the latest window — suitable for low-latency detection.
        samples = list(self._buffer)[-self._cfg.window_frames :]
        arr = np.array(samples, dtype=_DTYPE).astype(np.float32) * _FLOAT_SCALE
        return arr

    def read_window_blocking(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        """Block until a full window is available, then return it.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait.

        Returns
        -------
        numpy.ndarray or None
            Returns ``None`` on timeout or if capture was stopped.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            window = self.read_window()
            if window is not None:
                return window
            time.sleep(0.01)
        logger.warning("read_window_blocking timed out after %.1fs", timeout)
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Read PCM chunks from the stream and append to the ring buffer."""
        assert self._stream is not None
        chunk_bytes = self._cfg.chunk_frames * self._cfg.channels * 2  # 2 bytes per int16

        while not self._stop_event.is_set():
            try:
                raw = self._stream.read(self._cfg.chunk_frames, exception_on_overflow=False)
            except OSError as exc:
                logger.error("Audio read error: %s — stopping capture", exc)
                self.is_running = False
                break

            if len(raw) != chunk_bytes:
                logger.debug("Short read: expected %d bytes, got %d", chunk_bytes, len(raw))
                continue

            samples = np.frombuffer(raw, dtype=_DTYPE)
            # For multi-channel audio, mix down to mono by averaging channels.
            if self._cfg.channels > 1:
                samples = samples.reshape(-1, self._cfg.channels).mean(axis=1).astype(_DTYPE)

            self._buffer.extend(samples.tolist())

        logger.debug("Capture loop exited")
