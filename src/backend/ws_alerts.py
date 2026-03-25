"""
ws_alerts.py — WebSocket alert broadcast manager and ML socket ingestion.

Architecture
------------
AlertManager is a singleton that:
  1. Maintains a set of active WebSocket connections (FastAPI WebSocket objects).
  2. Keeps a ring-buffer of the last N alert events (configurable, default 50)
     for backfill when a new client connects.
  3. Runs a background asyncio task that reads JSON-newline events from the
     ML Unix domain socket and fans them out to all connected clients.

ML socket contract (agreed with ml-engineer, TASK-201/202):
  Each event is a UTF-8 JSON object terminated by a newline character (\n).
  Schema:
    {
      "type":       "cry" | "motion",
      "confidence": float,          // 0.0 – 1.0
      "timestamp":  "<ISO8601>"     // e.g. "2026-03-25T02:14:33.123Z"
    }

Staleness filter:
  Events older than settings.alert_max_age_seconds (default 30 s) are NOT
  delivered to newly connected clients — they would be confusing and are
  outside the safety-relevant window.  Live events are always forwarded
  regardless of age (they are by definition fresh).

Delivery guarantee:
  At-most-once per client.  We do not retry failed sends; if a WebSocket send
  raises, the client is removed from the active set and must reconnect.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data model                                                                   #
# --------------------------------------------------------------------------- #


class AlertEvent:
    """Thin wrapper around a raw alert dict with a parsed receive timestamp."""

    __slots__ = ("raw", "received_at")

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.received_at = datetime.now(tz=timezone.utc)

    def is_stale(self, max_age_seconds: int) -> bool:
        age = (datetime.now(tz=timezone.utc) - self.received_at).total_seconds()
        return age > max_age_seconds

    def to_json(self) -> str:
        return json.dumps(self.raw)


# --------------------------------------------------------------------------- #
# Alert manager                                                                #
# --------------------------------------------------------------------------- #


class AlertManager:
    """
    Singleton alert bus.

    Thread-safety note: all methods are called from within the single asyncio
    event loop that FastAPI/Uvicorn runs.  No locking is required as long as
    this invariant holds (which it does for a standard Uvicorn deployment).
    """

    def __init__(self, buffer_size: int = 50, max_age_seconds: int = 30) -> None:
        self._buffer_size = buffer_size
        self._max_age_seconds = max_age_seconds
        self._connections: set[WebSocket] = set()
        self._event_buffer: deque[AlertEvent] = deque(maxlen=buffer_size)
        self._reader_task: asyncio.Task[None] | None = None
        self._socket_path: str | None = None

    # ------------------------------------------------------------------ #
    # Connection lifecycle                                                 #
    # ------------------------------------------------------------------ #

    async def connect(self, websocket: WebSocket) -> None:
        """
        Accept a new WebSocket connection and send the buffered backfill.

        The backfill contains only events that are not yet stale, preserving
        the at-most-once / no-stale-events contract.
        """
        await websocket.accept()
        self._connections.add(websocket)
        logger.info(
            "WS client connected: %s (total=%d)",
            websocket.client,
            len(self._connections),
        )

        # Send non-stale buffered events as a backfill batch
        fresh_events = [
            e for e in self._event_buffer if not e.is_stale(self._max_age_seconds)
        ]
        for event in fresh_events:
            try:
                await websocket.send_text(event.to_json())
            except Exception as exc:
                logger.warning("Backfill send failed for %s: %s", websocket.client, exc)
                self._connections.discard(websocket)
                return

        logger.debug("Sent %d backfill events to %s", len(fresh_events), websocket.client)

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket from the active set (safe to call multiple times)."""
        self._connections.discard(websocket)
        logger.info(
            "WS client disconnected: %s (remaining=%d)",
            websocket.client,
            len(self._connections),
        )

    # ------------------------------------------------------------------ #
    # Broadcasting                                                         #
    # ------------------------------------------------------------------ #

    async def broadcast(self, event: AlertEvent) -> None:
        """
        Fan out a single AlertEvent to every connected client.

        Failed sends are caught individually; one bad client does not block
        others.  Disconnected clients are removed from the active set.
        """
        if not self._connections:
            return

        payload = event.to_json()
        dead: set[WebSocket] = set()

        for ws in list(self._connections):
            try:
                await ws.send_text(payload)
            except (WebSocketDisconnect, Exception) as exc:
                logger.debug("Broadcast send failed for %s: %s", ws.client, exc)
                dead.add(ws)

        for ws in dead:
            self.disconnect(ws)

    # ------------------------------------------------------------------ #
    # ML socket reader (background task)                                   #
    # ------------------------------------------------------------------ #

    def start_ml_reader(self, socket_path: str) -> None:
        """
        Spawn the background asyncio task that reads from the ML Unix socket.

        Safe to call multiple times — if a task is already running it is left
        intact.  Should be called from the FastAPI lifespan startup handler.
        """
        if self._reader_task is not None and not self._reader_task.done():
            logger.debug("ML reader task already running")
            return
        self._socket_path = socket_path
        self._reader_task = asyncio.create_task(
            self._ml_socket_reader_loop(socket_path),
            name="ml-socket-reader",
        )
        logger.info("ML socket reader task started (path=%s)", socket_path)

    def stop_ml_reader(self) -> None:
        """Cancel the background reader task.  Called from lifespan shutdown."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            logger.info("ML socket reader task cancelled")

    async def _ml_socket_reader_loop(self, socket_path: str) -> None:
        """
        Continuously connect to the ML Unix domain socket, read newline-delimited
        JSON events, buffer them, and broadcast to connected WebSocket clients.

        Reconnection strategy:
          - On any connection or read error, wait 2 seconds then retry.
          - This loop never exits except when cancelled.
        """
        reconnect_delay = 2.0  # seconds between reconnect attempts

        while True:
            reader: asyncio.StreamReader | None = None
            writer: asyncio.StreamWriter | None = None

            try:
                reader, writer = await asyncio.open_unix_connection(socket_path)
                logger.info("Connected to ML socket at %s", socket_path)

                async for line in reader:
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        continue

                    try:
                        raw = json.loads(line_str)
                    except json.JSONDecodeError as exc:
                        logger.warning("Malformed ML event (skipping): %s | raw=%r", exc, line_str)
                        continue

                    # Validate minimum required fields
                    if not _is_valid_alert(raw):
                        logger.warning("Invalid alert schema (skipping): %r", raw)
                        continue

                    event = AlertEvent(raw)
                    self._event_buffer.append(event)
                    await self.broadcast(event)

            except asyncio.CancelledError:
                logger.info("ML socket reader: cancelled")
                break
            except FileNotFoundError:
                logger.debug(
                    "ML socket %s not found — ML pipeline may not be running yet. "
                    "Retrying in %.1fs",
                    socket_path,
                    reconnect_delay,
                )
            except ConnectionRefusedError:
                logger.debug(
                    "ML socket connection refused — retrying in %.1fs", reconnect_delay
                )
            except Exception as exc:
                logger.error(
                    "ML socket reader error: %s — retrying in %.1fs",
                    exc,
                    reconnect_delay,
                    exc_info=True,
                )
            finally:
                if writer is not None:
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass

            # Back-off before reconnect (skip if cancelled)
            try:
                await asyncio.sleep(reconnect_delay)
            except asyncio.CancelledError:
                break

    # ------------------------------------------------------------------ #
    # Introspection                                                         #
    # ------------------------------------------------------------------ #

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    @property
    def buffered_event_count(self) -> int:
        return len(self._event_buffer)

    def recent_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return raw dicts for the most recent buffered events (newest last)."""
        events = list(self._event_buffer)
        if limit is not None:
            events = events[-limit:]
        return [e.raw for e in events]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _is_valid_alert(raw: Any) -> bool:
    """Return True if raw conforms to the ML alert schema."""
    if not isinstance(raw, dict):
        return False
    if raw.get("type") not in ("cry", "motion"):
        return False
    if not isinstance(raw.get("confidence"), (int, float)):
        return False
    if not isinstance(raw.get("timestamp"), str):
        return False
    return True


# --------------------------------------------------------------------------- #
# Module-level singleton                                                       #
# --------------------------------------------------------------------------- #

# Imported and used by main.py
alert_manager = AlertManager()
