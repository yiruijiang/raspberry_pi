"""
socket_publisher.py — Unix domain socket server for ML detection event delivery.

Protocol
--------
The publisher binds a Unix domain socket and accepts one persistent client
connection at a time (the backend process).  Each detection event is written as
a single UTF-8 JSON line followed by a newline character (``\\n``), so the
backend can use a simple ``readline()`` loop.

Wire format (one line per event):
    {"type": "cry", "confidence": 0.9312, "timestamp": "2026-03-25T14:32:01.123Z"}\\n

This matches the spec schema:
    { "type": "cry" | "motion", "confidence": float, "timestamp": ISO8601 }

Connection lifecycle
--------------------
1. Publisher creates and binds the socket on :meth:`SocketPublisher.start`.
2. A background thread accepts a client connection and stores it.
3. :meth:`SocketPublisher.publish` serialises the event and sends it to the
   connected client.  If no client is connected, the event is silently dropped
   (detection should not block on socket I/O).
4. If the client disconnects, the background thread re-enters accept() and
   waits for the backend to reconnect — no process restart required.

Thread safety
-------------
:meth:`publish` is called from the inference thread.  The socket send is
protected by a lock so that concurrent calls (e.g., from a future motion
detection thread) cannot interleave partial writes.

Backpressure
------------
The socket send uses a non-blocking write with a short timeout.  If the backend
is too slow to consume events, the oldest unread data in the kernel buffer will
be overwritten; detection events are ephemeral and dropping an old event is
preferable to blocking inference.

Usage
-----
    from socket_publisher import SocketPublisher
    from config import config

    pub = SocketPublisher(config.socket)
    pub.start()

    try:
        pub.publish(event.to_dict())
    finally:
        pub.stop()
"""

import json
import logging
import os
import socket
import threading
from typing import Optional

from config import SocketConfig

logger = logging.getLogger(__name__)

# Maximum bytes allowed to accumulate in the kernel send buffer before we
# consider the client stalled and close the connection.
_SEND_TIMEOUT_SECONDS = 0.1
_RECV_BUFFER_SIZE = 4096  # not used for reading, but required for socket setup


class SocketPublisher:
    """Publishes ML detection events over a Unix domain socket.

    Parameters
    ----------
    config:
        :class:`~config.SocketConfig` providing ``socket_path``.
    """

    def __init__(self, config: SocketConfig) -> None:
        self._cfg = config
        self._server_socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None
        self._client_lock = threading.Lock()
        self._accept_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.is_running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bind the Unix socket and start accepting connections.

        Removes any stale socket file from a previous run before binding.

        Raises
        ------
        OSError
            If the socket cannot be created or bound.
        """
        if self.is_running:
            logger.warning("SocketPublisher.start() called while already running")
            return

        socket_path = self._cfg.socket_path
        self._remove_stale_socket(socket_path)

        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._server_socket.bind(socket_path)
        except OSError as exc:
            self._server_socket.close()
            self._server_socket = None
            raise OSError(f"Cannot bind Unix socket at '{socket_path}': {exc}") from exc

        self._server_socket.listen(1)
        # Non-blocking accept with timeout so the accept thread can check the
        # stop event regularly.
        self._server_socket.settimeout(1.0)

        self._stop_event.clear()
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="socket-accept",
            daemon=True,
        )
        self.is_running = True
        self._accept_thread.start()
        logger.info("SocketPublisher listening at %s", socket_path)

    def stop(self) -> None:
        """Stop accepting connections and close all sockets."""
        if not self.is_running:
            return

        self._stop_event.set()

        with self._client_lock:
            if self._client_socket is not None:
                try:
                    self._client_socket.close()
                except Exception:
                    pass
                self._client_socket = None

        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None

        if self._accept_thread is not None:
            self._accept_thread.join(timeout=3.0)
            self._accept_thread = None

        self._remove_stale_socket(self._cfg.socket_path)
        self.is_running = False
        logger.info("SocketPublisher stopped")

    def publish(self, event: dict) -> bool:
        """Serialise and send a detection event to the connected backend client.

        Parameters
        ----------
        event:
            Dict conforming to the spec schema, e.g.::

                {"type": "cry", "confidence": 0.93, "timestamp": "..."}

        Returns
        -------
        bool
            ``True`` if the event was sent successfully, ``False`` if no client
            is connected or the send failed (event silently dropped).
        """
        line = json.dumps(event, separators=(",", ":")) + "\n"
        data = line.encode("utf-8")

        with self._client_lock:
            if self._client_socket is None:
                logger.debug(
                    "No backend client connected — dropping event type=%s", event.get("type")
                )
                return False

            try:
                self._client_socket.settimeout(_SEND_TIMEOUT_SECONDS)
                self._client_socket.sendall(data)
                logger.debug("Published event: %s", line.rstrip())
                return True
            except (BrokenPipeError, ConnectionResetError):
                logger.warning(
                    "Backend client disconnected during send — will re-accept"
                )
                self._client_socket.close()
                self._client_socket = None
                return False
            except socket.timeout:
                logger.warning(
                    "Send timeout (%.1fs) — backend client is not consuming events fast enough; "
                    "dropping event",
                    _SEND_TIMEOUT_SECONDS,
                )
                return False
            except OSError as exc:
                logger.error("Socket send error: %s", exc)
                self._client_socket.close()
                self._client_socket = None
                return False

    @property
    def has_client(self) -> bool:
        """True if a backend client is currently connected."""
        with self._client_lock:
            return self._client_socket is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _accept_loop(self) -> None:
        """Block waiting for a client connection; re-enter on disconnect."""
        while not self._stop_event.is_set():
            try:
                client_sock, _ = self._server_socket.accept()
            except socket.timeout:
                # Check stop_event and loop.
                continue
            except OSError:
                # Server socket closed — exit.
                break

            client_sock.settimeout(_SEND_TIMEOUT_SECONDS)
            with self._client_lock:
                if self._client_socket is not None:
                    # Reject second connection — only one consumer expected.
                    logger.warning(
                        "Refusing second connection on ML socket "
                        "(only one backend consumer supported)"
                    )
                    client_sock.close()
                    continue
                self._client_socket = client_sock

            logger.info("Backend client connected to ML socket")

            # Wait until the client disconnects (detect via recv returning empty).
            self._wait_for_disconnect(client_sock)

            with self._client_lock:
                if self._client_socket is client_sock:
                    self._client_socket = None

            logger.info("Backend client disconnected from ML socket — waiting for reconnect")

        logger.debug("Accept loop exited")

    def _wait_for_disconnect(self, client_sock: socket.socket) -> None:
        """Block until the client closes its end of the connection."""
        client_sock.settimeout(1.0)
        while not self._stop_event.is_set():
            try:
                data = client_sock.recv(_RECV_BUFFER_SIZE)
                if not data:
                    # Orderly shutdown by client.
                    break
                # The backend should not be sending us data, but ignore it.
            except socket.timeout:
                continue
            except OSError:
                break

    @staticmethod
    def _remove_stale_socket(path: str) -> None:
        """Remove a leftover socket file from a previous process run."""
        try:
            if os.path.exists(path):
                os.unlink(path)
                logger.debug("Removed stale socket file at %s", path)
        except OSError as exc:
            logger.warning("Could not remove stale socket at %s: %s", path, exc)
