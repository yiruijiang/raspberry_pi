import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      // Workbox generates the service worker at build time.
      // During dev, the SW is not active so hot-reload works normally.
      workbox: {
        // Cache the app shell (HTML, JS, CSS) for offline access.
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}'],
        // Do not cache API or stream URLs — those must always be live.
        navigateFallback: '/index.html',
        navigateFallbackDenylist: [/^\/v1\//, /^\/ws\//],
        runtimeCaching: [
          {
            // Cache static assets aggressively
            urlPattern: /\.(js|css|woff2|png|ico|svg)$/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'static-assets',
              expiration: { maxAgeSeconds: 60 * 60 * 24 * 7 }, // 7 days
            },
          },
        ],
      },
      manifest: {
        name: 'Baby Monitor',
        short_name: 'BabyMon',
        description: 'Live baby monitor — video feed and smart alerts',
        theme_color: '#1a1a2e',
        background_color: '#0d0d1a',
        display: 'standalone',
        orientation: 'portrait-primary',
        start_url: '/',
        icons: [
          {
            src: '/icons/icon-192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any maskable',
          },
          {
            src: '/icons/icon-512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable',
          },
        ],
      },
    }),
  ],
  server: {
    // In dev, proxy API and WebSocket requests to the Pi backend so we
    // avoid CORS issues and don't need to hardcode the Pi address in source.
    // Set VITE_API_BASE_URL in .env.local to override for production.
    proxy: {
      '/v1': {
        target: process.env.VITE_BACKEND_ORIGIN ?? 'http://raspberrypi.local:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: process.env.VITE_BACKEND_ORIGIN ?? 'http://raspberrypi.local:8000',
        changeOrigin: true,
        ws: true,
      },
      '/healthz': {
        target: process.env.VITE_BACKEND_ORIGIN ?? 'http://raspberrypi.local:8000',
        changeOrigin: true,
      },
    },
  },
});
