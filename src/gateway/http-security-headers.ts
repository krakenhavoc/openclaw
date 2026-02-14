import type { ServerResponse } from "node:http";

/**
 * Apply security headers to all HTTP responses from the gateway.
 * Call this at the top of `handleRequest` before any route dispatch.
 */
export function applyGlobalSecurityHeaders(res: ServerResponse): void {
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("X-Frame-Options", "DENY");
  res.setHeader("Content-Security-Policy", "frame-ancestors 'none'");
  res.setHeader("Referrer-Policy", "no-referrer");
}

/**
 * Apply HSTS header (only when TLS is active on the server).
 */
export function applyHstsHeader(res: ServerResponse): void {
  res.setHeader("Strict-Transport-Security", "max-age=31536000; includeSubDomains");
}
