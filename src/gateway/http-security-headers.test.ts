import { describe, expect, it } from "vitest";
import type { ServerResponse } from "node:http";
import { applyGlobalSecurityHeaders, applyHstsHeader } from "./http-security-headers.js";

function createMockResponse(): ServerResponse & { _headers: Record<string, string> } {
  const headers: Record<string, string> = {};
  return {
    _headers: headers,
    setHeader(name: string, value: string) {
      headers[name.toLowerCase()] = value;
    },
  } as ServerResponse & { _headers: Record<string, string> };
}

describe("applyGlobalSecurityHeaders", () => {
  it("sets all required security headers", () => {
    const res = createMockResponse();
    applyGlobalSecurityHeaders(res);

    expect(res._headers["x-content-type-options"]).toBe("nosniff");
    expect(res._headers["x-frame-options"]).toBe("DENY");
    expect(res._headers["content-security-policy"]).toBe("frame-ancestors 'none'");
    expect(res._headers["referrer-policy"]).toBe("no-referrer");
  });

  it("does not set HSTS header", () => {
    const res = createMockResponse();
    applyGlobalSecurityHeaders(res);

    expect(res._headers["strict-transport-security"]).toBeUndefined();
  });
});

describe("applyHstsHeader", () => {
  it("sets Strict-Transport-Security header", () => {
    const res = createMockResponse();
    applyHstsHeader(res);

    expect(res._headers["strict-transport-security"]).toBe(
      "max-age=31536000; includeSubDomains",
    );
  });
});
