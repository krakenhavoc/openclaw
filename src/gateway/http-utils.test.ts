import type { IncomingMessage } from "node:http";
import { describe, expect, it } from "vitest";
import { resolveSessionKey } from "./http-utils.js";

function fakeReq(headers: Record<string, string> = {}): IncomingMessage {
  return { headers } as unknown as IncomingMessage;
}

describe("resolveSessionKey", () => {
  it("generates a session key when no override header is present", () => {
    const key = resolveSessionKey({
      req: fakeReq(),
      agentId: "main",
      prefix: "test",
    });
    expect(key).toBeTruthy();
    expect(typeof key).toBe("string");
  });

  it("returns null when override header is present but not allowed", () => {
    const key = resolveSessionKey({
      req: fakeReq({ "x-openclaw-session-key": "custom-key" }),
      agentId: "main",
      prefix: "test",
      allowOverride: false,
    });
    expect(key).toBeNull();
  });

  it("returns null when override header is present and allowOverride is undefined", () => {
    const key = resolveSessionKey({
      req: fakeReq({ "x-openclaw-session-key": "custom-key" }),
      agentId: "main",
      prefix: "test",
    });
    expect(key).toBeNull();
  });

  it("allows override when allowOverride is true", () => {
    const key = resolveSessionKey({
      req: fakeReq({ "x-openclaw-session-key": "custom-key" }),
      agentId: "main",
      prefix: "test",
      allowOverride: true,
    });
    expect(key).toBe("custom-key");
  });

  it("uses user-based key when user is provided and no override", () => {
    const key = resolveSessionKey({
      req: fakeReq(),
      agentId: "main",
      user: "alice",
      prefix: "openai",
    });
    expect(key).toContain("openai-user:alice");
  });
});
