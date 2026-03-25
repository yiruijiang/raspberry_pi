/**
 * hooks/useStreamToken.ts
 *
 * Encapsulates the stream token lifecycle:
 *   1. Fetch a stream token from POST /v1/auth/stream-token using the Bearer JWT.
 *   2. Schedule a refresh 50 seconds after each successful fetch (10s before
 *      the 60s TTL expires), updating the token in state so callers can rebuild
 *      the stream URL without manual intervention.
 *   3. Clean up the timer on unmount or when the bearer token changes.
 *
 * TASK-105 is now deployed. The real POST /v1/auth/stream-token endpoint is
 * wired below. No mock fallback — if the endpoint is unreachable, the hook
 * enters 'error' status and VideoStream surfaces a disconnected overlay.
 *
 * Token is held in React state only — never written to localStorage or
 * sessionStorage.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchStreamToken } from '../api/client';

/** How many seconds before expiry to request a fresh token. */
const REFRESH_BEFORE_EXPIRY_S = 10;

/** Fallback TTL to use if the response does not include expires_in. */
const DEFAULT_TTL_S = 60;

export type StreamTokenStatus =
  | 'idle'        // not yet started
  | 'fetching'    // in-flight request
  | 'ready'       // token available
  | 'refreshing'  // background refresh in progress (token still valid)
  | 'error';      // fetch failed

export interface UseStreamTokenResult {
  /** Current signed stream token, or null if not yet available. */
  streamToken: string | null;
  /** Lifecycle status for UI feedback. */
  status: StreamTokenStatus;
  /** Last error detail string, if status === 'error'. */
  error: string | null;
  /** Imperatively trigger a token refresh (e.g., after stream error). */
  refresh: () => void;
}

/**
 * @param bearerToken - The JWT obtained from POST /v1/auth/token.
 *   Pass null to disable fetching (e.g., user not yet logged in).
 */
export function useStreamToken(bearerToken: string | null): UseStreamTokenResult {
  const [streamToken, setStreamToken] = useState<string | null>(null);
  const [status, setStatus] = useState<StreamTokenStatus>('idle');
  const [error, setError] = useState<string | null>(null);

  // Keep a ref to the refresh timer so we can cancel it on re-fetch or unmount.
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Track whether this effect invocation is still current (prevents setting
  // state after unmount or after bearer token changes).
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const doFetch = useCallback(
    async (isRefresh: boolean) => {
      if (!bearerToken) return;

      setStatus(isRefresh ? 'refreshing' : 'fetching');
      setError(null);

      try {
        // Call the real POST /v1/auth/stream-token endpoint (TASK-105 deployed).
        // Contract: requires Authorization: Bearer <jwt>, returns
        // { stream_token: string, expires_in: number } where expires_in is an
        // integer (60 in the current backend config).
        const response = await fetchStreamToken(bearerToken);

        if (!mountedRef.current) return;

        setStreamToken(response.stream_token);
        setStatus('ready');

        // Schedule the next refresh before the token expires.
        const ttl = response.expires_in ?? DEFAULT_TTL_S;
        const refreshIn = Math.max(0, ttl - REFRESH_BEFORE_EXPIRY_S) * 1000;

        clearTimer();
        timerRef.current = setTimeout(() => {
          if (mountedRef.current) {
            void doFetch(true);
          }
        }, refreshIn);
      } catch (err: unknown) {
        if (!mountedRef.current) return;

        const message =
          err instanceof Error ? err.message : 'Unknown error fetching stream token';
        setError(message);
        setStatus('error');
        // Do not schedule a retry here — error recovery is handled at a higher
        // level (VideoStream retries on img error, callers can call refresh()).
      }
    },
    [bearerToken, clearTimer],
  );

  // Expose an imperative refresh handle for callers (e.g., VideoStream after
  // an img error event triggers the retry flow).
  const refresh = useCallback(() => {
    clearTimer();
    void doFetch(false);
  }, [clearTimer, doFetch]);

  // Fetch on mount and whenever the bearer token changes.
  useEffect(() => {
    if (!bearerToken) {
      setStreamToken(null);
      setStatus('idle');
      setError(null);
      clearTimer();
      return;
    }

    void doFetch(false);

    // Cleanup: cancel any pending refresh timer when the token changes or
    // the component using this hook unmounts.
    return () => {
      clearTimer();
    };
  }, [bearerToken, doFetch, clearTimer]);

  return { streamToken, status, error, refresh };
}
