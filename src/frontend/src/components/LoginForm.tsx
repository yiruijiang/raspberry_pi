/**
 * components/LoginForm.tsx
 *
 * Simple login form that exchanges HTTP Basic credentials for a Bearer JWT
 * via POST /v1/auth/token (openapi.yaml — issueToken).
 *
 * The JWT is held in parent state (App.tsx) and passed down — never stored
 * in localStorage or sessionStorage per TASK-103 acceptance criteria.
 */

import React, { useState, useCallback } from 'react';
import { issueToken, ApiError } from '../api/client';
import styles from './LoginForm.module.css';

interface LoginFormProps {
  onSuccess: (bearerToken: string) => void;
}

export const LoginForm: React.FC<LoginFormProps> = ({ onSuccess }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!username || !password) return;

      setIsLoading(true);
      setError(null);

      try {
        const res = await issueToken(username, password);
        onSuccess(res.access_token);
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          setError('Incorrect username or password.');
        } else {
          setError('Could not reach the monitor. Check that the Pi is on.');
        }
      } finally {
        setIsLoading(false);
      }
    },
    [username, password, onSuccess],
  );

  return (
    <main className={styles.page} aria-label="Login">
      <form
        className={styles.form}
        onSubmit={(e) => void handleSubmit(e)}
        aria-label="Sign in to baby monitor"
        noValidate
      >
        <h1 className={styles.heading}>Baby Monitor</h1>
        <p className={styles.subheading}>Sign in to view the live feed</p>

        {error && (
          <div
            className={styles.errorBanner}
            role="alert"
            aria-live="assertive"
          >
            {error}
          </div>
        )}

        <label className={styles.label} htmlFor="username">
          Username
        </label>
        <input
          id="username"
          className={styles.input}
          type="text"
          value={username}
          autoComplete="username"
          autoCapitalize="none"
          spellCheck={false}
          disabled={isLoading}
          onChange={(e) => setUsername(e.target.value)}
          required
        />

        <label className={styles.label} htmlFor="password">
          Password
        </label>
        <input
          id="password"
          className={styles.input}
          type="password"
          value={password}
          autoComplete="current-password"
          disabled={isLoading}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        <button
          className={styles.submitButton}
          type="submit"
          disabled={isLoading || !username || !password}
          aria-busy={isLoading}
        >
          {isLoading ? 'Signing in...' : 'Sign in'}
        </button>
      </form>
    </main>
  );
};
