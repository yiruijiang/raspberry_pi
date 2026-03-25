/**
 * components/ErrorBoundary.tsx
 *
 * React class-based error boundary. Wraps major UI sections so that a
 * JavaScript error in one section (e.g., video stream, alert panel) does
 * not crash the entire dashboard.
 *
 * Usage:
 *   <ErrorBoundary label="Video feed">
 *     <VideoStream ... />
 *   </ErrorBoundary>
 */

import React, { Component, type ReactNode } from 'react';
import styles from './ErrorBoundary.module.css';

interface ErrorBoundaryProps {
  children: ReactNode;
  /** Human-readable label for the section — shown in the fallback UI. */
  label?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  message: string | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, message: null };
  }

  static getDerivedStateFromError(error: unknown): ErrorBoundaryState {
    const message =
      error instanceof Error ? error.message : 'An unexpected error occurred.';
    return { hasError: true, message };
  }

  componentDidCatch(error: unknown, info: React.ErrorInfo) {
    // In production the Pi has no external error tracker, so log to console.
    console.error('[ErrorBoundary]', this.props.label ?? 'section', error, info);
  }

  handleReset = () => {
    this.setState({ hasError: false, message: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          className={styles.container}
          role="alert"
          aria-label={`Error in ${this.props.label ?? 'section'}`}
        >
          <p className={styles.heading}>
            {this.props.label ? `${this.props.label} error` : 'Something went wrong'}
          </p>
          {this.state.message && (
            <p className={styles.detail}>{this.state.message}</p>
          )}
          <button
            className={styles.retryButton}
            onClick={this.handleReset}
            type="button"
          >
            Retry
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
