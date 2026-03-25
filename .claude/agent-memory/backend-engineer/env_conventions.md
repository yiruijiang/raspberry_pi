---
name: Environment variable conventions
description: All backend env var names, types, defaults, and secret management rules
type: project
---

All configuration is read via pydantic-settings from environment variables (case-insensitive) or a `.env` file in the working directory. No secrets are hardcoded.

Template: `src/backend/.env.example` — always keep in sync with `config.py`.

Key variables:
- `JWT_SECRET` — HS256 signing secret, min 32 chars. MUST be set in production. Default warns.
- `JWT_ALGORITHM` — "HS256" (default) or "RS256"
- `JWT_EXPIRE_MINUTES` — default 720 (12h)
- `MONITOR_USERNAME` / `MONITOR_PASSWORD` — Basic auth credentials for token issuance
- `ML_SOCKET_PATH` — default `/tmp/baby_monitor_ml.sock`
- `TLS_CERTFILE` / `TLS_KEYFILE` — PEM paths; leave blank for dev HTTP
- `CAMERA_INDEX`, `CAMERA_WIDTH`, `CAMERA_HEIGHT`, `CAMERA_FPS`, `JPEG_QUALITY`
- `AUDIO_DEVICE_INDEX` (None = system default), `AUDIO_SAMPLE_RATE`, `AUDIO_CHANNELS`, `AUDIO_CHUNK_FRAMES`
- `ALERT_BUFFER_SIZE` (default 50), `ALERT_MAX_AGE_SECONDS` (default 30)
- `STREAM_TOKEN_SECRET` — HMAC key for stream tokens; falls back to JWT_SECRET if unset. Set independently to allow rotation without invalidating JWTs.
- `STREAM_TOKEN_TTL_SECONDS` — stream token validity window, default 60, min 10, max 3600
- `CORS_ORIGINS` — comma-separated, default "*" (set to Pi IP in production)

**Rules:**
- Never log `JWT_SECRET`, `MONITOR_PASSWORD`, or any token value.
- `.env` is gitignored; `.env.example` is committed.
- In tests, call `get_settings.cache_clear()` before and after each test that patches env vars.
