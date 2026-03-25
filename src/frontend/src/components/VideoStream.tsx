/**
 * components/VideoStream.tsx
 *
 * Renders the live MJPEG video feed from the Pi camera.
 *
 * ── Stream token lifecycle ────────────────────────────────────────────────────
 * The stream endpoint is JWT-protected via signed URL tokens (ADR-001).
 * Browsers cannot attach Authorization headers to <img src> requests, so the
 * token is appended as a query parameter: /v1/stream/video?stream_token=<token>.
 *
 * Token refresh is handled by useStreamToken (50s interval, 10s before the
 * 60s TTL). Updating <img src> causes the browser to close the old MJPEG
 * connection and open a new one — this is the intended reconnect mechanism.
 *
 * ── Error / retry logic ───────────────────────────────────────────────────────
 * On <img> onerror: wait 2s, fetch a fresh token, retry.
 * After MAX_RETRIES (5) consecutive failures: enter 'disconnected' state and
 * stop retrying until the parent resets via the key prop or remount.
 *
 * ── iOS Safari note ──────────────────────────────────────────────────────────
 * MJPEG (multipart/x-mixed-replace) rendering on iOS Safari 16+ has historically
 * been inconsistent and must be verified on a real device before production
 * release. A real-device test on iOS Safari 16+ is REQUIRED before TASK-103
 * can be considered production-ready.
 *
 * In the meantime, this component detects iOS Safari at runtime and renders a
 * non-blocking advisory banner so users understand why the stream may not appear,
 * without crashing or silently showing a blank area.
 *
 * If real-device testing confirms MJPEG is broken on iOS Safari, escalate to PM:
 * that is a scope change requiring a WebRTC or HLS fallback and cannot be
 * resolved as a frontend-only fix within the current task definition.
 *
 * ── Props ─────────────────────────────────────────────────────────────────────
 * @prop bearerToken  - JWT for stream-token auth. Pass null while loading.
 * @prop className    - Optional CSS class for the wrapper element.
 * @prop onDisconnect - Called when MAX_RETRIES is exceeded.
 * @prop onConnected  - Called when the first frame loads successfully.
 */

import React, {
  useRef,
  useEffect,
  useState,
  useCallback,
  useMemo,
} from 'react';
import { useStreamToken } from '../hooks/useStreamToken';
import { buildVideoStreamUrl } from '../api/client';
import styles from './VideoStream.module.css';

const MAX_RETRIES = 5;
const RETRY_DELAY_MS = 2_000;

export type VideoStreamStatus =
  | 'idle'
  | 'connecting'
  | 'live'
  | 'reconnecting'
  | 'disconnected';

interface VideoStreamProps {
  bearerToken: string | null;
  className?: string;
  onDisconnect?: () => void;
  onConnected?: () => void;
}

// ── iOS Safari detection ──────────────────────────────────────────────────────
// Detects iOS Safari by user-agent. We intentionally do this at render time
// (not in a useEffect) so the warning is present on the first paint — no flash.
// This covers iPhone and iPad in both Safari and WKWebView (UIWebView is gone
// in iOS 15+). Chrome/Firefox on iOS use WKWebView but report different UA
// strings; their MJPEG support may also vary, so we flag all iOS browsers.
//
// NOTE: A real-device test on iOS Safari 16+ is REQUIRED before production
// release. If the stream renders correctly, this banner can be removed.
function detectIos(): boolean {
  if (typeof navigator === 'undefined') return false;
  // iPadOS 13+ reports a Mac UA in desktop mode; check maxTouchPoints as well.
  const ua = navigator.userAgent;
  const isIphone = /iPhone|iPod/.test(ua);
  const isIpad = /iPad/.test(ua) || (
    /Macintosh/.test(ua) && navigator.maxTouchPoints > 1
  );
  return isIphone || isIpad;
}

export const VideoStream: React.FC<VideoStreamProps> = ({
  bearerToken,
  className,
  onDisconnect,
  onConnected,
}) => {
  const { streamToken, status: tokenStatus, refresh: refreshToken } =
    useStreamToken(bearerToken);

  // The src string set on the <img> element. Null means the img is not yet mounted.
  const [imgSrc, setImgSrc] = useState<string | null>(null);

  // Tracks how many consecutive errors have occurred without a successful load.
  const retryCountRef = useRef(0);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Stream status drives the status overlay.
  const [streamStatus, setStreamStatus] = useState<VideoStreamStatus>('idle');

  // iOS detection is stable for the lifetime of the component — no need to
  // re-compute on re-render. useMemo with an empty dep array evaluates once.
  const isIos = useMemo(() => detectIos(), []);

  // ── Apply the stream token whenever it changes ───────────────────────────────
  // When useStreamToken delivers a new token (initial fetch or 50s refresh),
  // set the img src. The browser closes the old MJPEG connection and opens a
  // new one. This also resets the retry counter since we have a fresh token.
  useEffect(() => {
    if (streamToken) {
      retryCountRef.current = 0; // fresh token — reset consecutive error count
      setImgSrc(buildVideoStreamUrl(streamToken));
      setStreamStatus('connecting');
    }
  }, [streamToken]);

  // ── Reflect token-fetch state in stream status ───────────────────────────────
  useEffect(() => {
    if (tokenStatus === 'error') {
      setStreamStatus('disconnected');
    }
  }, [tokenStatus]);

  // ── Cleanup retry timer on unmount ──────────────────────────────────────────
  useEffect(() => {
    return () => {
      if (retryTimerRef.current !== null) {
        clearTimeout(retryTimerRef.current);
      }
    };
  }, []);

  // ── <img> event handlers ─────────────────────────────────────────────────────

  const handleLoad = useCallback(() => {
    // First frame has arrived — stream is live.
    retryCountRef.current = 0;
    setStreamStatus('live');
    onConnected?.();
  }, [onConnected]);

  const handleError = useCallback(() => {
    const currentRetry = retryCountRef.current + 1;
    retryCountRef.current = currentRetry;

    if (currentRetry >= MAX_RETRIES) {
      setStreamStatus('disconnected');
      setImgSrc(null); // unmount the img to stop any browser retry attempts
      onDisconnect?.();
      return;
    }

    setStreamStatus('reconnecting');

    // Clear any pending retry before scheduling a new one.
    if (retryTimerRef.current !== null) {
      clearTimeout(retryTimerRef.current);
    }

    retryTimerRef.current = setTimeout(() => {
      // Request a fresh stream token — the 401 that triggered the error may
      // mean the previous token expired early. refreshToken() will update
      // streamToken in state, which triggers the useEffect above to update imgSrc.
      refreshToken();
    }, RETRY_DELAY_MS);
  }, [refreshToken, onDisconnect]);

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div
      className={[styles.container, className].filter(Boolean).join(' ')}
      role="region"
      aria-label="Live video feed"
      aria-live="polite"
    >
      {/* iOS Safari advisory banner — rendered above the feed, non-blocking.
          Shown only on iOS devices. If real-device testing (iOS Safari 16+)
          confirms MJPEG works reliably, remove this banner. */}
      {isIos && (
        <div
          className={styles.iosSafariWarning}
          role="status"
          aria-label="iOS Safari compatibility notice"
        >
          <span className={styles.iosSafariWarningIcon} aria-hidden="true">
            {/* unicode information symbol — no emoji */}
            &#9432;
          </span>
          <span className={styles.iosSafariWarningText}>
            Live video may not be supported on iOS Safari. If the stream
            doesn&apos;t appear, please use Chrome or the desktop browser.
          </span>
        </div>
      )}

      {/* The MJPEG stream element. Only mounted when we have a src. */}
      {imgSrc && (
        <img
          className={styles.feed}
          src={imgSrc}
          alt="Live baby monitor feed"
          onLoad={handleLoad}
          onError={handleError}
          // Disable browser-level drag so accidental swipes don't navigate away.
          draggable={false}
        />
      )}

      {/* Status overlay — shown when stream is not live */}
      {streamStatus !== 'live' && (
        <div
          className={styles.overlay}
          aria-hidden={streamStatus === 'live'}
        >
          <StatusIndicator status={streamStatus} />
        </div>
      )}

      {/* Screen-reader live region for status changes */}
      <span className="sr-only" aria-live="assertive" aria-atomic="true">
        {streamStatus === 'disconnected'
          ? 'Video feed disconnected. Please refresh the page.'
          : streamStatus === 'reconnecting'
          ? 'Video feed reconnecting...'
          : ''}
      </span>
    </div>
  );
};

// ── StatusIndicator sub-component ─────────────────────────────────────────────

interface StatusIndicatorProps {
  status: VideoStreamStatus;
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({ status }) => {
  const messages: Record<VideoStreamStatus, string> = {
    idle:         'Waiting for stream...',
    connecting:   'Connecting...',
    live:         'Live',
    reconnecting: 'Reconnecting...',
    disconnected: 'Feed disconnected',
  };

  return (
    <div className={styles.statusIndicator} data-status={status}>
      <span className={styles.statusDot} aria-hidden="true" />
      <span className={styles.statusText}>{messages[status]}</span>
    </div>
  );
};
