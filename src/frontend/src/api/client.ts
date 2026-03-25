/**
 * api/client.ts
 *
 * Typed API client for all backend endpoints.
 *
 * Contract source: src/backend/docs/openapi.yaml (authoritative)
 * Auth decision:   docs/specs/decisions/ADR-001-video-stream-auth.md
 *
 * All network URLs are constructed relative to `VITE_API_BASE_URL`
 * (default: empty string, meaning same-origin / proxied via Vite dev server).
 * Set VITE_API_BASE_URL=http://raspberrypi.local:8000 for production builds
 * served from a different origin than the API.
 *
 * No credentials are stored in localStorage or sessionStorage.
 * All tokens live in React state / closure memory only.
 */

// ── Base URL ──────────────────────────────────────────────────────────────────
// In dev, Vite proxies /v1/* and /ws/* to the Pi, so base is empty.
// In production, set VITE_API_BASE_URL in the build environment.
const API_BASE: string = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '';

// ── Types matching openapi.yaml schemas ───────────────────────────────────────

export interface TokenResponse {
  access_token: string;
  token_type: 'bearer';
  /** Validity in seconds — returned as a string by the backend for broad compat */
  expires_in: string;
}

/**
 * Stream token response from POST /v1/auth/stream-token.
 *
 * NOTE: This endpoint is defined in the product spec and ADR-001 but is NOT yet
 * present in openapi.yaml as of 2026-03-25. It is being added by the backend
 * engineer in TASK-105. The shape below matches what is specified in ADR-001 and
 * TASK-103. Do not change without coordinating with the backend engineer.
 */
export interface StreamTokenResponse {
  stream_token: string;
  /** TTL in seconds; currently always 60 */
  expires_in: number;
}

export type AlertType = 'cry' | 'motion';

export interface AlertEvent {
  type: AlertType;
  /** Model confidence score 0–1 */
  confidence: number;
  /** ISO 8601 UTC timestamp */
  timestamp: string;
}

export interface RecentAlertsResponse {
  events: AlertEvent[];
  total_buffered: number;
}

export interface StreamStatusResponse {
  camera: {
    available: boolean;
    backend: 'picamera2' | 'opencv' | null;
    resolution: string;
    fps: number;
  };
  audio: {
    available: boolean;
    backend: 'pyaudio' | null;
    sample_rate: number;
    channels: number;
    detail?: string;
  };
  ws_clients: number;
}

export interface ErrorResponse {
  detail: string;
}

// ── Error class ───────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string,
  ) {
    super(`API ${status}: ${detail}`);
    this.name = 'ApiError';
  }
}

// ── Internal fetch helper ─────────────────────────────────────────────────────

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init.headers,
    },
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as ErrorResponse;
      detail = body.detail ?? detail;
    } catch {
      // Non-JSON error body — use statusText
    }
    throw new ApiError(res.status, detail);
  }

  // Some endpoints return no body (204), guard against JSON parse error
  const text = await res.text();
  return text ? (JSON.parse(text) as T) : ({} as T);
}

// ── Auth endpoints ────────────────────────────────────────────────────────────

/**
 * POST /v1/auth/token
 *
 * Exchange HTTP Basic credentials for a Bearer JWT.
 * Returns a TokenResponse. Throws ApiError on 401 (bad credentials).
 */
export async function issueToken(
  username: string,
  password: string,
): Promise<TokenResponse> {
  const credentials = btoa(`${username}:${password}`);
  return request<TokenResponse>('/v1/auth/token', {
    method: 'POST',
    headers: {
      Authorization: `Basic ${credentials}`,
    },
  });
}

/**
 * POST /v1/auth/stream-token
 *
 * Exchange a Bearer JWT for a short-lived (60s) signed stream token
 * suitable for use as a URL query parameter on the MJPEG stream endpoint.
 *
 * DEPENDENCY: Requires TASK-105 (backend-engineer) to be deployed.
 * Until TASK-105 is live, this function will receive a 404. The
 * useStreamToken hook handles this via a DEV_MOCK_TOKEN fallback
 * (see hooks/useStreamToken.ts).
 */
export async function fetchStreamToken(
  bearerToken: string,
): Promise<StreamTokenResponse> {
  return request<StreamTokenResponse>('/v1/auth/stream-token', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${bearerToken}`,
    },
  });
}

// ── Stream endpoints ──────────────────────────────────────────────────────────

/**
 * Builds the authenticated MJPEG video stream URL.
 * The token is appended as a query parameter per ADR-001.
 * Never set <img src> to the bare stream URL without a token.
 */
export function buildVideoStreamUrl(streamToken: string): string {
  return `${API_BASE}/v1/stream/video?stream_token=${encodeURIComponent(streamToken)}`;
}

/**
 * Builds the authenticated audio stream URL.
 * Same token pattern as video (ADR-001 applies to both stream endpoints).
 */
export function buildAudioStreamUrl(streamToken: string): string {
  return `${API_BASE}/v1/stream/audio?stream_token=${encodeURIComponent(streamToken)}`;
}

/**
 * GET /v1/stream/status
 *
 * Returns camera/mic availability and connected WS client count.
 * Frontend polls this at 5-second intervals to drive the health badge.
 */
export async function fetchStreamStatus(
  bearerToken: string,
): Promise<StreamStatusResponse> {
  return request<StreamStatusResponse>('/v1/stream/status', {
    headers: { Authorization: `Bearer ${bearerToken}` },
  });
}

// ── Alert endpoints ───────────────────────────────────────────────────────────

/**
 * GET /v1/alerts/recent?limit=50
 *
 * Page-load hydration: fetch buffered alert events before WS is connected.
 * Max 50 events; backend filters to last 30 seconds on WS connect, but
 * REST endpoint returns whatever is in the buffer.
 */
export async function fetchRecentAlerts(
  bearerToken: string,
  limit = 50,
): Promise<RecentAlertsResponse> {
  return request<RecentAlertsResponse>(
    `/v1/alerts/recent?limit=${Math.min(limit, 50)}`,
    { headers: { Authorization: `Bearer ${bearerToken}` } },
  );
}

// ── WebSocket URL builder ─────────────────────────────────────────────────────

/**
 * Builds the WebSocket URL for /ws/alerts.
 *
 * The browser WebSocket API cannot set custom headers, so the JWT is passed
 * as a query parameter (?token=<jwt>), which the backend accepts per
 * openapi.yaml /ws/alerts spec.
 *
 * In dev, Vite proxies ws:// through to the Pi (ws: true in vite.config.ts).
 * In production, set VITE_API_BASE_URL to override the origin.
 */
export function buildAlertsWsUrl(bearerToken: string): string {
  // If API_BASE is empty (same-origin or proxied), derive the WS URL from
  // the current window location. If API_BASE is set (cross-origin), swap
  // the protocol explicitly.
  if (!API_BASE) {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${window.location.host}/ws/alerts?token=${encodeURIComponent(bearerToken)}`;
  }

  const wsBase = API_BASE.replace(/^http/, 'ws');
  return `${wsBase}/ws/alerts?token=${encodeURIComponent(bearerToken)}`;
}
