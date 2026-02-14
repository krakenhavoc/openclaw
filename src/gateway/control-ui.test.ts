import type { IncomingMessage, ServerResponse } from "node:http";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { handleControlUiHttpRequest } from "./control-ui.js";

const makeResponse = (): {
  res: ServerResponse;
  setHeader: ReturnType<typeof vi.fn>;
  end: ReturnType<typeof vi.fn>;
} => {
  const setHeader = vi.fn();
  const end = vi.fn();
  const res = {
    headersSent: false,
    statusCode: 200,
    setHeader,
    end,
  } as unknown as ServerResponse;
  return { res, setHeader, end };
};

describe("handleControlUiHttpRequest", () => {
  it("serves index.html with correct content-type and cache headers", async () => {
    const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-ui-"));
    try {
      await fs.writeFile(path.join(tmp, "index.html"), "<html></html>\n");
      const { res, setHeader } = makeResponse();
      const handled = handleControlUiHttpRequest(
        { url: "/", method: "GET" } as IncomingMessage,
        res,
        {
          root: { kind: "resolved", path: tmp },
        },
      );
      expect(handled).toBe(true);
      // Security headers (X-Frame-Options, CSP) are now applied globally in
      // server-http.ts via applyGlobalSecurityHeaders(), not per-handler.
      expect(setHeader).toHaveBeenCalledWith("Content-Type", "text/html; charset=utf-8");
      expect(setHeader).toHaveBeenCalledWith("Cache-Control", "no-cache");
    } finally {
      await fs.rm(tmp, { recursive: true, force: true });
    }
  });
});
