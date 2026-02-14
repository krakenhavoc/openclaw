import { describe, expect, it } from "vitest";
import type { IncomingMessage } from "node:http";
import { resolveSessionKey } from "./http-utils.js";

function createMockRequest(headers: Record<string, string> = {}): IncomingMessage {
  return {
    headers: Object.fromEntries(
      Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v]),
    ),
  } as IncomingMessage;
}

describe("resolveSessionKey", () => {
  it("generates a session key when no override header is present", () => {
    const req = createMockRequest();
    const key = resolveSessionKey({ req, agentId: "main", prefix: "openai" });
    expect(key).toBeTruthy();
    expect(typeof key).toBe("string");
  });

  it("returns null when override header is present and allowOverride is false", () => {
    const req = createMockRequest({
      "x-openclaw-session-key": "custom:key:123",
    });
    const key = resolveSessionKey({
      req,
      agentId: "main",
      prefix: "openai",
      allowOverride: false,
    });
    expect(key).toBeNull();
  });

  it("returns null when override header is present and allowOverride is undefined (default)", () => {
    const req = createMockRequest({
      "x-openclaw-session-key": "custom:key:123",
    });
    const key = resolveSessionKey({
      req,
      agentId: "main",
      prefix: "openai",
    });
    expect(key).toBeNull();
  });

  it("returns the explicit key when allowOverride is true", () => {
    const req = createMockRequest({
      "x-openclaw-session-key": "custom:key:123",
    });
    const key = resolveSessionKey({
      req,
      agentId: "main",
      prefix: "openai",
      allowOverride: true,
    });
    expect(key).toBe("custom:key:123");
  });

  it("ignores empty/whitespace-only override header", () => {
    const req = createMockRequest({
      "x-openclaw-session-key": "   ",
    });
    const key = resolveSessionKey({
      req,
      agentId: "main",
      prefix: "openai",
      allowOverride: false,
    });
    // Empty header is treated as absent — generates a key
    expect(key).toBeTruthy();
    expect(typeof key).toBe("string");
  });

  it("includes user in generated session key", () => {
    const req = createMockRequest();
    const key = resolveSessionKey({
      req,
      agentId: "main",
      prefix: "openai",
      user: "alice",
    });
    expect(key).toContain("openai-user:alice");
  });
});
