/**
 * components/AlertBanner.tsx
 *
 * Displays real-time cry and motion alerts delivered via the /ws/alerts
 * WebSocket connection (managed by useAlerts hook).
 *
 * Layout (mobile-first):
 *  - Banner at the top of the dashboard for the most recent unacknowledged alert.
 *  - Alert history list below — newest first, scrollable.
 *  - Each alert shows: type label, confidence percentage, relative time, dismiss.
 *
 * Accessibility:
 *  - New alerts are announced via an aria-live="assertive" region.
 *  - Dismiss button has a minimum 44×44px touch target.
 *  - Alert type is communicated via text, not color alone.
 */

import React, { useCallback } from 'react';
import { useAlerts } from '../hooks/useAlerts';
import type { AlertEvent } from '../api/client';
import styles from './AlertBanner.module.css';

interface AlertBannerProps {
  bearerToken: string | null;
  /** Max history entries to show (default 10 — full 50 is available in history view). */
  maxVisible?: number;
}

export const AlertBanner: React.FC<AlertBannerProps> = ({
  bearerToken,
  maxVisible = 10,
}) => {
  const { alerts, connectionStatus, clearAlerts } = useAlerts(bearerToken);

  const visibleAlerts = alerts.slice(0, maxVisible);
  const mostRecent = alerts[0] ?? null;

  // Connection status badge label
  const wsStatusLabel: Record<typeof connectionStatus, string> = {
    idle:         'Alerts: offline',
    connecting:   'Alerts: connecting',
    connected:    'Alerts: live',
    reconnecting: 'Alerts: reconnecting',
    disconnected: 'Alerts: disconnected',
  };

  return (
    <section
      className={styles.container}
      aria-label="Alert panel"
    >
      {/* ── Screen-reader live region ──────────────────────────────── */}
      {/* Assertive so screen readers interrupt current speech for urgent alerts */}
      <div
        aria-live="assertive"
        aria-atomic="true"
        className="sr-only"
      >
        {mostRecent
          ? `${mostRecent.type === 'cry' ? 'Cry detected' : 'Motion detected'}, ` +
            `confidence ${Math.round(mostRecent.confidence * 100)}%, ` +
            `at ${formatRelativeTime(mostRecent.timestamp)}.`
          : ''}
      </div>

      {/* ── Header row ────────────────────────────────────────────── */}
      <div className={styles.header}>
        <h2 className={styles.title}>Alerts</h2>
        <span
          className={styles.wsStatus}
          data-status={connectionStatus}
          aria-label={wsStatusLabel[connectionStatus]}
          title={wsStatusLabel[connectionStatus]}
        >
          <span className={styles.wsDot} aria-hidden="true" />
          <span className={styles.wsLabel}>{wsStatusLabel[connectionStatus]}</span>
        </span>
        {alerts.length > 0 && (
          <button
            className={`${styles.clearButton} touch-target`}
            onClick={clearAlerts}
            aria-label="Clear all alerts"
            type="button"
          >
            Clear
          </button>
        )}
      </div>

      {/* ── Alert list ────────────────────────────────────────────── */}
      {visibleAlerts.length === 0 ? (
        <p className={styles.empty} aria-live="polite">
          No recent alerts.
        </p>
      ) : (
        <ul className={styles.alertList} role="list" aria-label="Recent alerts">
          {visibleAlerts.map((alert) => (
            <AlertItem key={`${alert.type}-${alert.timestamp}`} alert={alert} />
          ))}
        </ul>
      )}

      {/* ── Overflow indicator ────────────────────────────────────── */}
      {alerts.length > maxVisible && (
        <p className={styles.overflow} aria-live="polite">
          +{alerts.length - maxVisible} older alerts not shown
        </p>
      )}
    </section>
  );
};

// ── AlertItem sub-component ───────────────────────────────────────────────────

interface AlertItemProps {
  alert: AlertEvent;
}

const AlertItem: React.FC<AlertItemProps> = ({ alert }) => {
  const isCry = alert.type === 'cry';
  const confidencePct = Math.round(alert.confidence * 100);
  const relativeTime = formatRelativeTime(alert.timestamp);

  return (
    <li
      className={styles.alertItem}
      data-type={alert.type}
      // Each item is a landmark so assistive tech can navigate alerts by item.
      role="listitem"
    >
      {/* Type icon (text-based, not icon-only, for accessibility) */}
      <span className={styles.alertIcon} aria-hidden="true">
        {isCry ? '🔊' : '👁'}
      </span>

      <div className={styles.alertBody}>
        <span className={styles.alertType}>
          {isCry ? 'Cry detected' : 'Motion detected'}
        </span>
        <span className={styles.alertMeta}>
          <span
            className={styles.confidence}
            aria-label={`${confidencePct}% confidence`}
          >
            {confidencePct}%
          </span>
          <span className={styles.separator} aria-hidden="true">·</span>
          <time
            className={styles.alertTime}
            dateTime={alert.timestamp}
            title={new Date(alert.timestamp).toLocaleString()}
          >
            {relativeTime}
          </time>
        </span>
      </div>

      {/* Confidence bar — visual reinforcement, aria-hidden since % is text */}
      <div
        className={styles.confidenceBar}
        aria-hidden="true"
        style={{ '--confidence': `${confidencePct}%` } as React.CSSProperties}
      />
    </li>
  );
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Format a timestamp as a short relative string, e.g. "just now", "2m ago".
 * Uses absolute time for events older than 1 hour.
 */
function formatRelativeTime(isoTimestamp: string): string {
  const now = Date.now();
  const then = new Date(isoTimestamp).getTime();
  const diffS = Math.floor((now - then) / 1000);

  if (diffS < 10) return 'just now';
  if (diffS < 60) return `${diffS}s ago`;
  if (diffS < 3600) return `${Math.floor(diffS / 60)}m ago`;
  return new Date(isoTimestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
}
