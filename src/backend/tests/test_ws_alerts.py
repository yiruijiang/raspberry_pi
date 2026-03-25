"""
test_ws_alerts.py — Unit tests for ws_alerts.py.

Tests cover AlertManager connection lifecycle, backfill, broadcast,
staleness filtering, and the event validation helper.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.ws_alerts import AlertEvent, AlertManager, _is_valid_alert


# ── AlertEvent ──────────────────────────────────────────────────────────────

def test_alert_event_not_stale_when_fresh():
    raw = {"type": "cry", "confidence": 0.9, "timestamp": "2026-03-25T00:00:00Z"}
    event = AlertEvent(raw)
    assert not event.is_stale(30)


def test_alert_event_stale_when_old():
    raw = {"type": "motion", "confidence": 0.5, "timestamp": "2026-03-25T00:00:00Z"}
    event = AlertEvent(raw)
    # Manually backdating received_at
    event.received_at = datetime.now(tz=timezone.utc) - timedelta(seconds=60)
    assert event.is_stale(30)


def test_alert_event_to_json_round_trips():
    raw = {"type": "cry", "confidence": 0.87, "timestamp": "2026-03-25T01:00:00Z"}
    event = AlertEvent(raw)
    decoded = json.loads(event.to_json())
    assert decoded == raw


# ── _is_valid_alert ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ({"type": "cry",    "confidence": 0.9,  "timestamp": "2026-03-25T00:00:00Z"}, True),
    ({"type": "motion", "confidence": 0.1,  "timestamp": "2026-03-25T00:00:00Z"}, True),
    ({"type": "unknown","confidence": 0.9,  "timestamp": "2026-03-25T00:00:00Z"}, False),
    ({"type": "cry",    "confidence": "hi", "timestamp": "2026-03-25T00:00:00Z"}, False),
    ({"type": "cry",    "confidence": 0.9                                       }, False),
    ("not a dict",                                                                  False),
    ({},                                                                            False),
])
def test_is_valid_alert(raw, expected):
    assert _is_valid_alert(raw) == expected


# ── AlertManager ────────────────────────────────────────────────────────────

def make_mock_websocket():
    ws = AsyncMock()
    ws.client = ("127.0.0.1", 12345)
    return ws


@pytest.mark.asyncio
async def test_connect_adds_to_connections():
    mgr = AlertManager(buffer_size=50, max_age_seconds=30)
    ws = make_mock_websocket()
    await mgr.connect(ws)
    assert mgr.connection_count == 1
    ws.accept.assert_awaited_once()


@pytest.mark.asyncio
async def test_disconnect_removes_from_connections():
    mgr = AlertManager()
    ws = make_mock_websocket()
    await mgr.connect(ws)
    mgr.disconnect(ws)
    assert mgr.connection_count == 0


@pytest.mark.asyncio
async def test_disconnect_idempotent():
    mgr = AlertManager()
    ws = make_mock_websocket()
    mgr.disconnect(ws)   # should not raise even if ws was never connected
    assert mgr.connection_count == 0


@pytest.mark.asyncio
async def test_backfill_sends_fresh_events():
    mgr = AlertManager(max_age_seconds=30)
    raw = {"type": "cry", "confidence": 0.9, "timestamp": "2026-03-25T00:00:00Z"}
    mgr._event_buffer.append(AlertEvent(raw))

    ws = make_mock_websocket()
    await mgr.connect(ws)

    # One backfill send (the buffered event)
    ws.send_text.assert_awaited_once_with(json.dumps(raw))


@pytest.mark.asyncio
async def test_backfill_skips_stale_events():
    mgr = AlertManager(max_age_seconds=30)
    raw = {"type": "cry", "confidence": 0.9, "timestamp": "2026-03-25T00:00:00Z"}
    event = AlertEvent(raw)
    event.received_at = datetime.now(tz=timezone.utc) - timedelta(seconds=60)
    mgr._event_buffer.append(event)

    ws = make_mock_websocket()
    await mgr.connect(ws)

    # Stale event must NOT be sent
    ws.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_broadcast_sends_to_all_connected():
    mgr = AlertManager()
    ws1 = make_mock_websocket()
    ws2 = make_mock_websocket()
    await mgr.connect(ws1)
    await mgr.connect(ws2)

    raw = {"type": "motion", "confidence": 0.6, "timestamp": "2026-03-25T00:01:00Z"}
    await mgr.broadcast(AlertEvent(raw))

    payload = json.dumps(raw)
    ws1.send_text.assert_awaited_with(payload)
    ws2.send_text.assert_awaited_with(payload)


@pytest.mark.asyncio
async def test_broadcast_removes_dead_clients():
    from fastapi.websockets import WebSocketDisconnect
    mgr = AlertManager()
    ws = make_mock_websocket()
    ws.send_text.side_effect = WebSocketDisconnect()
    await mgr.connect(ws)

    await mgr.broadcast(AlertEvent({"type": "cry", "confidence": 0.9, "timestamp": "t"}))
    assert mgr.connection_count == 0


@pytest.mark.asyncio
async def test_broadcast_with_no_clients_does_not_raise():
    mgr = AlertManager()
    await mgr.broadcast(AlertEvent({"type": "cry", "confidence": 0.8, "timestamp": "t"}))
    # No assertion needed — just must not raise


def test_recent_events_returns_raw_dicts():
    mgr = AlertManager()
    raw = {"type": "cry", "confidence": 0.9, "timestamp": "2026-03-25T00:00:00Z"}
    mgr._event_buffer.append(AlertEvent(raw))
    events = mgr.recent_events(limit=10)
    assert events == [raw]


def test_event_buffer_respects_maxlen():
    mgr = AlertManager(buffer_size=3)
    for i in range(5):
        raw = {"type": "cry", "confidence": float(i) / 10, "timestamp": "t"}
        mgr._event_buffer.append(AlertEvent(raw))
    assert mgr.buffered_event_count == 3
