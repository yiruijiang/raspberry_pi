/**
 * App.tsx
 *
 * Root component. Manages auth state and renders the dashboard.
 *
 * Auth state machine:
 *   'unauthenticated' → user sees LoginForm
 *   'authenticated'   → user sees Dashboard (VideoStream + AlertBanner)
 *
 * The Bearer JWT is held in React state — not persisted to localStorage.
 * This means users must log in again after page refresh, which is acceptable
 * for the LAN-only use case and preferred for security hygiene.
 *
 * Error boundaries are applied around major sections so a crash in one panel
 * does not take down the whole dashboard.
 */

import React, { useState, useCallback } from 'react';
import { LoginForm } from './components/LoginForm';
import { VideoStream } from './components/VideoStream';
import { AlertBanner } from './components/AlertBanner';
import { ErrorBoundary } from './components/ErrorBoundary';
import styles from './App.module.css';

type AuthState = 'unauthenticated' | 'authenticated';

const App: React.FC = () => {
  const [authState, setAuthState] = useState<AuthState>('unauthenticated');
  // JWT held in memory only — never stored in localStorage/sessionStorage.
  const [bearerToken, setBearerToken] = useState<string | null>(null);

  const handleLoginSuccess = useCallback((token: string) => {
    setBearerToken(token);
    setAuthState('authenticated');
  }, []);

  const handleLogout = useCallback(() => {
    setBearerToken(null);
    setAuthState('unauthenticated');
  }, []);

  if (authState === 'unauthenticated') {
    return <LoginForm onSuccess={handleLoginSuccess} />;
  }

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <h1 className={styles.appTitle}>
          <span aria-hidden="true">◉</span> Baby Monitor
        </h1>
        <button
          className={`${styles.logoutButton} touch-target`}
          onClick={handleLogout}
          type="button"
          aria-label="Sign out"
        >
          Sign out
        </button>
      </header>

      <main className={styles.dashboard}>
        {/* ── Video feed ──────────────────────────────────────────── */}
        <ErrorBoundary label="Video feed">
          <section className={styles.feedSection} aria-label="Live feed">
            <VideoStream
              bearerToken={bearerToken}
              className={styles.videoStream}
            />
          </section>
        </ErrorBoundary>

        {/* ── Alert panel ─────────────────────────────────────────── */}
        <ErrorBoundary label="Alert panel">
          <AlertBanner
            bearerToken={bearerToken}
            maxVisible={10}
          />
        </ErrorBoundary>
      </main>
    </div>
  );
};

export default App;
