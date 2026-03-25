"""
stream.py — Pi camera capture and MJPEG / audio streaming.

Design decisions
----------------
- Video: Picamera2 (libcamera-based) is the recommended path on Pi OS Bookworm
  for CSI cameras.  We fall back to OpenCV / V4L2 when Picamera2 is not
  installed (e.g., on a dev laptop), allowing local development without a Pi.
- Container: MJPEG multipart/x-mixed-replace is the Phase 1 format.  It
  delivers sub-second latency on LAN, requires no client-side JS, and renders
  natively in <img> tags.  Each JPEG boundary is sent immediately — no segment
  accumulation overhead like HLS.
- Codec: JPEG baseline.  Broad hardware-decode support, no B-frame latency.
- Audio: PyAudio reads PCM frames from the USB mic and yields them as raw
  chunks.  The browser <audio> element can play raw PCM with MediaSource
  extensions; for simplicity we send raw signed 16-bit LE PCM, which is
  universally decodable via Web Audio API or a thin JS wrapper.
- Graceful degradation: if the camera or mic is unavailable, the generator
  raises a 503 StreamUnavailableError rather than crashing the server.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Generator

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Picamera2 / OpenCV import — runtime fallback for dev environments            #
# --------------------------------------------------------------------------- #
try:
    from picamera2 import Picamera2  # type: ignore[import-not-found]

    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False
    logger.warning(
        "picamera2 not found — falling back to OpenCV V4L2 capture. "
        "This is fine for local development; use Picamera2 on the Pi."
    )

try:
    import cv2  # type: ignore[import-not-found]

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import pyaudio  # type: ignore[import-not-found]

    _PYAUDIO_AVAILABLE = True
except ImportError:
    _PYAUDIO_AVAILABLE = False
    logger.warning(
        "pyaudio not found — audio streaming will return 503 when requested."
    )


# --------------------------------------------------------------------------- #
# Custom exception                                                             #
# --------------------------------------------------------------------------- #


class StreamUnavailableError(RuntimeError):
    """Raised when the camera or microphone cannot be opened."""


# --------------------------------------------------------------------------- #
# MJPEG video stream                                                           #
# --------------------------------------------------------------------------- #

MJPEG_BOUNDARY = b"--frame"
MJPEG_CONTENT_TYPE = "multipart/x-mixed-replace; boundary=frame"


def _encode_jpeg_cv2(frame_bgr, quality: int) -> bytes:
    """Encode an OpenCV BGR frame to JPEG bytes."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buf = cv2.imencode(".jpg", frame_bgr, encode_params)
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def generate_mjpeg_picamera2(
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    jpeg_quality: int,
) -> Generator[bytes, None, None]:
    """
    Capture frames via Picamera2 and yield MJPEG multipart chunks.

    Each yielded bytes value is one complete multipart part including the
    MIME boundary and Content-Type header, ready to be sent over HTTP.

    Raises StreamUnavailableError if the camera cannot be opened.
    """
    try:
        camera = Picamera2(camera_num=camera_index)
        config = camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": float(fps)},
        )
        camera.configure(config)
        camera.start()
    except Exception as exc:
        raise StreamUnavailableError(f"Picamera2 init failed: {exc}") from exc

    logger.info("Picamera2 stream started at %dx%d @%dfps", width, height, fps)

    try:
        frame_interval = 1.0 / fps
        while True:
            t0 = time.monotonic()

            # Capture RGB frame and encode to JPEG
            rgb_frame = camera.capture_array("main")

            # Picamera2 gives RGB; convert to BGR for consistent cv2 path
            # but here we use PIL/io path to avoid mandatory cv2 dependency
            try:
                from PIL import Image  # type: ignore[import-not-found]

                img = Image.fromarray(rgb_frame)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=jpeg_quality)
                jpeg_bytes = buf.getvalue()
            except ImportError:
                # PIL not available — try cv2
                import numpy as np  # type: ignore[import-not-found]

                bgr = rgb_frame[:, :, ::-1]  # RGB → BGR
                jpeg_bytes = _encode_jpeg_cv2(bgr, jpeg_quality)

            yield (
                MJPEG_BOUNDARY
                + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg_bytes)).encode()
                + b"\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )

            # Pace the loop to the target fps (best-effort; no busy-spin)
            elapsed = time.monotonic() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except GeneratorExit:
        logger.info("MJPEG generator: client disconnected (GeneratorExit)")
    except Exception as exc:
        logger.error("MJPEG generator error: %s", exc, exc_info=True)
        raise
    finally:
        try:
            camera.stop()
            camera.close()
        except Exception:
            pass
        logger.info("Picamera2 camera closed")


def generate_mjpeg_opencv(
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    jpeg_quality: int,
) -> Generator[bytes, None, None]:
    """
    Capture frames via OpenCV (V4L2) and yield MJPEG multipart chunks.

    Used as the development fallback when Picamera2 is not available.
    Raises StreamUnavailableError if the camera cannot be opened.
    """
    if not _CV2_AVAILABLE:
        raise StreamUnavailableError(
            "Neither picamera2 nor opencv-python is installed; cannot stream video."
        )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise StreamUnavailableError(
            f"OpenCV could not open camera index {camera_index}"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    logger.info("OpenCV V4L2 stream started at %dx%d @%dfps", width, height, fps)

    try:
        frame_interval = 1.0 / fps
        while True:
            t0 = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                logger.warning("cv2.VideoCapture.read() returned False — retrying")
                time.sleep(0.1)
                continue

            jpeg_bytes = _encode_jpeg_cv2(frame, jpeg_quality)

            yield (
                MJPEG_BOUNDARY
                + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg_bytes)).encode()
                + b"\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )

            elapsed = time.monotonic() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except GeneratorExit:
        logger.info("OpenCV MJPEG generator: client disconnected")
    except Exception as exc:
        logger.error("OpenCV MJPEG generator error: %s", exc, exc_info=True)
        raise
    finally:
        cap.release()
        logger.info("OpenCV camera released")


def get_video_stream_generator(
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    jpeg_quality: int,
) -> Generator[bytes, None, None]:
    """
    Return the appropriate MJPEG generator depending on available libraries.

    Priority: Picamera2 (Pi CSI) > OpenCV (V4L2 / webcam / dev laptop)
    Raises StreamUnavailableError if neither is available.
    """
    if _PICAMERA2_AVAILABLE:
        return generate_mjpeg_picamera2(camera_index, width, height, fps, jpeg_quality)
    if _CV2_AVAILABLE:
        return generate_mjpeg_opencv(camera_index, width, height, fps, jpeg_quality)
    raise StreamUnavailableError(
        "No camera backend available. "
        "Install picamera2 (Pi) or opencv-python (dev) and restart the server."
    )


# --------------------------------------------------------------------------- #
# Audio stream                                                                 #
# --------------------------------------------------------------------------- #

AUDIO_CONTENT_TYPE = "audio/L16;rate={sample_rate};channels={channels}"


def generate_audio_pcm(
    device_index: int | None,
    sample_rate: int,
    channels: int,
    chunk_frames: int,
) -> Generator[bytes, None, None]:
    """
    Capture audio via PyAudio and yield raw signed 16-bit LE PCM chunks.

    The HTTP response uses Transfer-Encoding: chunked so the browser receives
    a continuous byte stream.  The frontend uses the Web Audio API to decode
    and play it.

    Raises StreamUnavailableError if PyAudio is not installed or the device
    cannot be opened.
    """
    if not _PYAUDIO_AVAILABLE:
        raise StreamUnavailableError(
            "pyaudio is not installed — audio streaming is unavailable."
        )

    pa = pyaudio.PyAudio()

    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_frames,
            input_device_index=device_index,
        )
    except OSError as exc:
        pa.terminate()
        raise StreamUnavailableError(
            f"PyAudio could not open microphone (device_index={device_index}): {exc}"
        ) from exc

    logger.info(
        "Audio stream started: %dHz %dch device=%s",
        sample_rate,
        channels,
        device_index,
    )

    try:
        while True:
            try:
                data = stream.read(chunk_frames, exception_on_overflow=False)
            except OSError as exc:
                logger.error("Audio read error: %s", exc)
                break
            yield data

    except GeneratorExit:
        logger.info("Audio generator: client disconnected")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        logger.info("PyAudio stream closed")
