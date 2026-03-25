/**
 * hooks/useAlerts.ts
 *
 * Manages the WebSocket connection to WS /ws/alerts and maintains an ordered
 * alert event list in state.
 *
 * Behaviour:
 *  - Opens a WebSocket as soon as a bearerToken is available.
 *  - On connect: the server sends up to 50 buffered events from the last 30s
 *    as individual JSON text frames — these are appended to state immediately.
 *  - Live events: each new frame is parsed and prepended (newest first) with
 *    deduplication by timestamp+type.
 *  - On disconnect (clean or network drop): back-off reconnect starting at 1s,
 *    doubling up to a 30s cap.
 *  - Cleans up the WebSocket on bearer token change or unmount.
 *
 * The hook limits the in-memory event list to MAX_ALERTS items (newest kept)
 * to avoid unbounded growth during long monitoring sessions.
 *
 * Contract source: src/backend/docs/openapi.yaml — /ws/alerts
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { buildAlertsWsUrl, type AlertEvent } from '../api/client';

const MAX_ALERTS = 50;
const INITIAL_RECONNECT_DELAY_MS = 1_000;
const MAX_RECONNECT_DELAY_MS = 30_000;

export type AlertsConnectionStatus =
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'disconnected'; // gave up or no bearer token

export interface UseAlertsResult {
  alerts: AlertEvent[];
  connectionStatus: AlertsConnectionStatus;
  /** Clear the local alert list (does not affect the server buffer). */
  clearAlerts: () => void;
}

/**
 * @param bearerToken - JWT from POST /v1/auth/token.
 *   Pass null to stay disconnected (user not logged in).
 */
export function useAlerts(bearerToken: string | null): UseAlertsResult {
  const [alerts, setAlerts] = useState<AlertEvent[]>([]);
  const [connectionStatus, setConnectionStatus] =
    useState<AlertsConnectionStatus>('idle');

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY_MS);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const closeWs = useCallback(() => {
    if (wsRef.current) {
      // Remove all listeners before closing to prevent reconnect logic
      // firing when we intentionally close.
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onerror = null;
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const connect = useCallback(
    (token: string) => {
      if (!mountedRef.current) return;

      setConnectionStatus('connecting');
      const url = buildAlertsWsUrl(token);
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setConnectionStatus('connected');
        // Reset back-off on successful connection.
        reconnectDelayRef.current = INITIAL_RECONNECT_DELAY_MS;
      };

      ws.onmessage = (event: MessageEvent<string>) => {
        if (!mountedRef.current) return;

        let parsed: unknown;
        try {
          parsed = JSON.parse(event.data);
        } catch {
          console.warn('[useAlerts] Received non-JSON WebSocket message:', event.data);
          return;
        }

        // Basic runtime validation — the server schema is AlertEvent.
        if (
          typeof parsed !== 'object' ||
          parsed === null ||
          typeof (parsed as Record<string, unknown>).type !== 'string' ||
          typeof (parsed as Record<string, unknown>).confidence !== 'number' ||
          typeof (parsed as Record<string, unknown>).timestamp !== 'string'
        ) {
          console.warn('[useAlerts] Unexpected message shape:', parsed);
          return;
        }

        const alert = parsed as AlertEvent;

        setAlerts((prev) => {
          // Deduplicate: ignore if an event with the same type+timestamp exists.
          const isDuplicate = prev.some(
            (a) => a.timestamp === alert.timestamp && a.type === alert.type,
          );
          if (isDuplicate) return prev;

          // Prepend newest-first, then trim to MAX_ALERTS.
          return [alert, ...prev].slice(0, MAX_ALERTS);
        });
      };

      ws.onerror = () => {
        // onclose fires after onerror; let onclose handle the reconnect.
        // Logging here for observability.
        console.warn('[useAlerts] WebSocket error');
      };

      ws.onclose = (event: CloseEvent) => {
        if (!mountedRef.current) return;

        // Code 1000 = normal closure (we called close() intentionally — but
        // we strip listeners before calling close(), so this only fires for
        // server-initiated normal close or browser-initiated).
        if (event.code === 1008 || event.code === 4001) {
          // 1008 = policy violation (401/auth error from server).
          // 4001 = custom code the backend may use for auth failure.
          // Do not reconnect; surface as disconnected.
          setConnectionStatus('disconnected');
          return;
        }

        // All other codes: schedule a reconnect with exponential back-off.
        setConnectionStatus('reconnecting');
        const delay = reconnectDelayRef.current;
        reconnectDelayRef.current = Math.min(delay * 2, MAX_RECONNECT_DELAY_MS);

        reconnectTimerRef.current = setTimeout(() => {
          if (mountedRef.current) {
            connect(token);
          }
        }, delay);
      };
    },
    [closeWs],
  );

  // Open connection when bearer token becomes available; close and reopen on change.
  useEffect(() => {
    clearReconnectTimer();
    closeWs();

    if (!bearerToken) {
      setConnectionStatus('idle');
      return;
    }

    // Reset back-off whenever we get a fresh token (new login).
    reconnectDelayRef.current = INITIAL_RECONNECT_DELAY_MS;
    connect(bearerToken);

    return () => {
      clearReconnectTimer();
      closeWs();
    };
  }, [bearerToken, connect, closeWs, clearReconnectTimer]);

  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  return { alerts, connectionStatus, clearAlerts };
}
