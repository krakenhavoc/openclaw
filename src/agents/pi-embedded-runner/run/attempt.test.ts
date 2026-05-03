import { streamSimple } from "@mariozechner/pi-ai";
import { describe, expect, it, vi } from "vitest";
import type { OpenClawConfig } from "../../../config/types.openclaw.js";
import { appendBootstrapPromptWarning } from "../../bootstrap-budget.js";
import { SYSTEM_PROMPT_CACHE_BOUNDARY } from "../../system-prompt-cache-boundary.js";
import { buildAgentSystemPrompt } from "../../system-prompt.js";
import { isDeepSeekOnAzureFoundry } from "../model.js";
import {
  buildContextEnginePromptCacheInfo,
  buildAfterTurnRuntimeContext,
  buildAfterTurnRuntimeContextFromUsage,
  composeSystemPromptWithHookContext,
  decodeHtmlEntitiesInObject,
  applyEmbeddedAttemptToolsAllow,
  extractInlineToolCall,
  isPrimaryBootstrapRun,
  mergeOrphanedTrailingUserPrompt,
  normalizeMessagesForLlmBoundary,
  prependSystemPromptAddition,
  remapInjectedContextFilesToWorkspace,
  resetEmbeddedAgentBaseStreamFnCacheForTest,
  resolveEmbeddedAgentBaseStreamFn,
  resolveAttemptFsWorkspaceOnly,
  resolveEmbeddedAgentStreamFn,
  resolveUnknownToolGuardThreshold,
  shouldCreateBundleMcpRuntimeForAttempt,
  resolveAttemptToolPolicyMessageProvider,
  resolvePromptBuildHookResult,
  resolvePromptModeForSession,
  shouldStripBootstrapFromEmbeddedContext,
  shouldWarnOnOrphanedUserRepair,
  wrapStreamFnExtractDeepSeekInlineToolCalls,
  wrapStreamFnRepairMalformedToolCallArguments,
  wrapStreamFnSanitizeMalformedToolCalls,
  wrapStreamFnTrimToolCallNames,
} from "./attempt.js";
import { buildEmbeddedAttemptToolRunContext } from "./attempt.tool-run-context.js";

type FakeWrappedStream = {
  result: () => Promise<unknown>;
  [Symbol.asyncIterator]: () => AsyncIterator<unknown>;
};

function createFakeStream(params: {
  events: unknown[];
  resultMessage: unknown;
}): FakeWrappedStream {
  return {
    async result() {
      return params.resultMessage;
    },
    [Symbol.asyncIterator]() {
      return (async function* () {
        for (const event of params.events) {
          yield event;
        }
      })();
    },
  };
}

async function invokeWrappedTestStream(
  wrap: (
    baseFn: (...args: never[]) => unknown,
  ) => (...args: never[]) => FakeWrappedStream | Promise<FakeWrappedStream>,
  baseFn: (...args: never[]) => unknown,
): Promise<FakeWrappedStream> {
  const wrappedFn = wrap(baseFn);
  return await Promise.resolve(wrappedFn({} as never, {} as never, {} as never));
}

describe("applyEmbeddedAttemptToolsAllow", () => {
  it("keeps explicit toolsAllow authoritative after force-added tools are built", () => {
    const tools = [{ name: "exec" }, { name: "read" }, { name: "message" }];

    expect(
      applyEmbeddedAttemptToolsAllow(tools, ["exec", "read"]).map((tool) => tool.name),
    ).toEqual(["exec", "read"]);
  });

  it("normalizes explicit toolsAllow entries before filtering", () => {
    const tools = [{ name: "cron" }, { name: "read" }, { name: "message" }];

    expect(
      applyEmbeddedAttemptToolsAllow(tools, [" cron ", "READ"]).map((tool) => tool.name),
    ).toEqual(["cron", "read"]);
  });
});

describe("buildEmbeddedAttemptToolRunContext", () => {
  it("carries runtime toolsAllow into coding tool construction", () => {
    expect(
      buildEmbeddedAttemptToolRunContext({
        trigger: "manual",
        jobId: "job-1",
        memoryFlushWritePath: "memory/log.md",
        toolsAllow: ["memory_search", "memory_get"],
      }),
    ).toMatchObject({
      trigger: "manual",
      jobId: "job-1",
      memoryFlushWritePath: "memory/log.md",
      runtimeToolAllowlist: ["memory_search", "memory_get"],
    });
  });
});

describe("normalizeMessagesForLlmBoundary", () => {
  it("strips tool result details before provider conversion", () => {
    const input = [
      {
        role: "toolResult",
        toolCallId: "call_1",
        toolName: "exec",
        content: [{ type: "text", text: "visible output" }],
        details: { aggregated: "hidden diagnostics" },
        isError: false,
        timestamp: 1,
      },
    ];

    const output = normalizeMessagesForLlmBoundary(
      input as Parameters<typeof normalizeMessagesForLlmBoundary>[0],
    ) as Array<Record<string, unknown>>;

    expect(output[0]).not.toHaveProperty("details");
    expect(output[0]?.content).toEqual([{ type: "text", text: "visible output" }]);
    expect(input[0]).toHaveProperty("details");
  });

  it("keeps runtime-context transcript entries out of the LLM boundary", () => {
    const input = [
      {
        role: "user",
        content: [{ type: "text", text: "visible ask" }],
        timestamp: 1,
      },
      {
        role: "custom",
        customType: "openclaw.runtime-context",
        content: "secret runtime context",
        display: false,
        timestamp: 2,
      },
      {
        role: "custom",
        customType: "other-extension-context",
        content: "normal custom context",
        display: false,
        timestamp: 3,
      },
    ];

    const output = normalizeMessagesForLlmBoundary(
      input as Parameters<typeof normalizeMessagesForLlmBoundary>[0],
    ) as Array<Record<string, unknown>>;

    expect(output).toHaveLength(2);
    expect(output).not.toEqual(
      expect.arrayContaining([expect.objectContaining({ customType: "openclaw.runtime-context" })]),
    );
    expect(output).toEqual(
      expect.arrayContaining([expect.objectContaining({ customType: "other-extension-context" })]),
    );
  });
});

describe("shouldCreateBundleMcpRuntimeForAttempt", () => {
  it("skips bundle MCP when tools are disabled or unavailable", () => {
    expect(shouldCreateBundleMcpRuntimeForAttempt({ toolsEnabled: false })).toBe(false);
    expect(shouldCreateBundleMcpRuntimeForAttempt({ toolsEnabled: true, disableTools: true })).toBe(
      false,
    );
  });

  it("creates bundle MCP only when the allowlist can reach bundle MCP tool names", () => {
    expect(shouldCreateBundleMcpRuntimeForAttempt({ toolsEnabled: true })).toBe(true);
    expect(shouldCreateBundleMcpRuntimeForAttempt({ toolsEnabled: true, toolsAllow: [] })).toBe(
      true,
    );
    expect(
      shouldCreateBundleMcpRuntimeForAttempt({
        toolsEnabled: true,
        toolsAllow: ["memory_search", "memory_get"],
      }),
    ).toBe(false);
    expect(
      shouldCreateBundleMcpRuntimeForAttempt({
        toolsEnabled: true,
        toolsAllow: ["bundle-mcp"],
      }),
    ).toBe(true);
    expect(
      shouldCreateBundleMcpRuntimeForAttempt({
        toolsEnabled: true,
        toolsAllow: ["strict__strict_probe"],
      }),
    ).toBe(true);
  });
});

describe("resolveAttemptToolPolicyMessageProvider", () => {
  it("prefers explicit tool-policy provider over transport channel", () => {
    expect(
      resolveAttemptToolPolicyMessageProvider({
        messageChannel: "discord",
        messageProvider: "discord-voice",
      }),
    ).toBe("discord-voice");
  });

  it("falls back to message channel when provider is omitted", () => {
    expect(resolveAttemptToolPolicyMessageProvider({ messageChannel: "discord" })).toBe("discord");
  });
});

describe("resolvePromptBuildHookResult", () => {
  function createLegacyOnlyHookRunner() {
    return {
      hasHooks: vi.fn(
        (
          hookName:
            | "agent_turn_prepare"
            | "heartbeat_prompt_contribution"
            | "before_prompt_build"
            | "before_agent_start",
        ) => hookName === "before_agent_start",
      ),
      runBeforePromptBuild: vi.fn(async () => undefined),
      runBeforeAgentStart: vi.fn(async () => ({ prependContext: "from-hook" })),
    };
  }

  it("reuses precomputed legacy before_agent_start result without invoking hook again", async () => {
    const hookRunner = createLegacyOnlyHookRunner();
    const result = await resolvePromptBuildHookResult({
      config: {},
      prompt: "hello",
      messages: [],
      hookCtx: {},
      hookRunner,
      legacyBeforeAgentStartResult: { prependContext: "from-cache", systemPrompt: "legacy-system" },
    });

    expect(hookRunner.runBeforeAgentStart).not.toHaveBeenCalled();
    expect(result).toEqual({
      prependContext: "from-cache",
      appendContext: undefined,
      systemPrompt: "legacy-system",
      prependSystemContext: undefined,
      appendSystemContext: undefined,
    });
  });

  it("calls legacy hook when precomputed result is absent", async () => {
    const hookRunner = createLegacyOnlyHookRunner();
    const messages = [{ role: "user", content: "ctx" }];
    const result = await resolvePromptBuildHookResult({
      config: {},
      prompt: "hello",
      messages,
      hookCtx: {},
      hookRunner,
    });

    expect(hookRunner.runBeforeAgentStart).toHaveBeenCalledTimes(1);
    expect(hookRunner.runBeforeAgentStart).toHaveBeenCalledWith({ prompt: "hello", messages }, {});
    expect(result.prependContext).toBe("from-hook");
  });

  it("merges prompt-build and legacy context fields in deterministic order", async () => {
    const hookRunner = {
      hasHooks: vi.fn(() => true),
      runBeforePromptBuild: vi.fn(async () => ({
        prependContext: "prompt context",
        appendContext: "prompt append context",
        prependSystemContext: "prompt prepend",
        appendSystemContext: "prompt append",
      })),
      runBeforeAgentStart: vi.fn(async () => ({
        prependContext: "legacy context",
        appendContext: "legacy append context",
        prependSystemContext: "legacy prepend",
        appendSystemContext: "legacy append",
      })),
    };

    const result = await resolvePromptBuildHookResult({
      config: {},
      prompt: "hello",
      messages: [],
      hookCtx: {},
      hookRunner,
    });

    expect(result.prependContext).toBe("prompt context\n\nlegacy context");
    expect(result.appendContext).toBe("prompt append context\n\nlegacy append context");
    expect(result.prependSystemContext).toBe("prompt prepend\n\nlegacy prepend");
    expect(result.appendSystemContext).toBe("prompt append\n\nlegacy append");
  });

  it("applies heartbeat prompt contributions only during heartbeat turns", async () => {
    const hookRunner = {
      hasHooks: vi.fn((hookName: string) => hookName === "heartbeat_prompt_contribution"),
      runHeartbeatPromptContribution: vi.fn(async () => ({
        prependContext: "heartbeat prepend",
        appendContext: "heartbeat append",
      })),
      runBeforePromptBuild: vi.fn(async () => undefined),
      runBeforeAgentStart: vi.fn(async () => undefined),
    };

    const heartbeatResult = await resolvePromptBuildHookResult({
      config: {},
      prompt: "hello",
      messages: [],
      hookCtx: { trigger: "heartbeat", sessionKey: "agent:main:main" },
      hookRunner,
    });

    expect(hookRunner.runHeartbeatPromptContribution).toHaveBeenCalledTimes(1);
    expect(heartbeatResult.prependContext).toBe("heartbeat prepend");
    expect(heartbeatResult.appendContext).toBe("heartbeat append");

    hookRunner.runHeartbeatPromptContribution.mockClear();
    const userResult = await resolvePromptBuildHookResult({
      config: {},
      prompt: "hello",
      messages: [],
      hookCtx: { trigger: "user", sessionKey: "agent:main:main" },
      hookRunner,
    });

    expect(hookRunner.runHeartbeatPromptContribution).not.toHaveBeenCalled();
    expect(userResult.prependContext).toBeUndefined();
    expect(userResult.appendContext).toBeUndefined();
  });
});

describe("composeSystemPromptWithHookContext", () => {
  it("returns undefined when no hook system context is provided", () => {
    expect(composeSystemPromptWithHookContext({ baseSystemPrompt: "base" })).toBeUndefined();
  });

  it("builds prepend/base/append system prompt order", () => {
    expect(
      composeSystemPromptWithHookContext({
        baseSystemPrompt: "  base system  ",
        prependSystemContext: "  prepend  ",
        appendSystemContext: "  append  ",
      }),
    ).toBe("prepend\n\nbase system\n\nappend");
  });

  it("normalizes hook system context line endings and trailing whitespace", () => {
    expect(
      composeSystemPromptWithHookContext({
        baseSystemPrompt: "  base system  ",
        prependSystemContext: "  prepend line  \r\nsecond line\t\r\n",
        appendSystemContext: "  append  \t\r\n",
      }),
    ).toBe("prepend line\nsecond line\n\nbase system\n\nappend");
  });

  it("avoids blank separators when base system prompt is empty", () => {
    expect(
      composeSystemPromptWithHookContext({
        baseSystemPrompt: "   ",
        appendSystemContext: "  append only  ",
      }),
    ).toBe("append only");
  });

  it("keeps hook-composed system prompt stable when bootstrap warnings only change the user prompt", () => {
    const baseSystemPrompt = buildAgentSystemPrompt({
      workspaceDir: "/tmp/openclaw",
      contextFiles: [{ path: "AGENTS.md", content: "Follow AGENTS guidance." }],
      toolNames: ["read"],
    });
    const composedSystemPrompt = composeSystemPromptWithHookContext({
      baseSystemPrompt,
      appendSystemContext: "hook system context",
    });
    const turns = [
      {
        systemPrompt: composedSystemPrompt,
        prompt: appendBootstrapPromptWarning("hello", ["AGENTS.md: 200 raw -> 0 injected"]),
      },
      {
        systemPrompt: composedSystemPrompt,
        prompt: appendBootstrapPromptWarning("hello again", []),
      },
      {
        systemPrompt: composedSystemPrompt,
        prompt: appendBootstrapPromptWarning("hello once more", [
          "AGENTS.md: 200 raw -> 0 injected",
        ]),
      },
    ];

    expect(turns[0]?.systemPrompt).toBe(turns[1]?.systemPrompt);
    expect(turns[1]?.systemPrompt).toBe(turns[2]?.systemPrompt);
    expect(turns[0]?.prompt.startsWith("hello")).toBe(true);
    expect(turns[1]?.prompt).toBe("hello again");
    expect(turns[2]?.prompt.startsWith("hello once more")).toBe(true);
    expect(turns[0]?.prompt).toContain("[Bootstrap truncation warning]");
    expect(turns[2]?.prompt).toContain("[Bootstrap truncation warning]");
  });
});

describe("resolvePromptModeForSession", () => {
  it("uses minimal mode for subagent sessions", () => {
    expect(resolvePromptModeForSession("agent:main:subagent:child")).toBe("minimal");
  });

  it("uses minimal mode for cron sessions", () => {
    expect(resolvePromptModeForSession("agent:main:cron:job-1")).toBe("minimal");
    expect(resolvePromptModeForSession("agent:main:cron:job-1:run:run-abc")).toBe("minimal");
  });

  it("uses full mode for regular and undefined sessions", () => {
    expect(resolvePromptModeForSession(undefined)).toBe("full");
    expect(resolvePromptModeForSession("agent:main")).toBe("full");
    expect(resolvePromptModeForSession("agent:main:thread:abc")).toBe("full");
  });
});

describe("shouldStripBootstrapFromEmbeddedContext", () => {
  it("never injects raw BOOTSTRAP.md into embedded system context", () => {
    expect(shouldStripBootstrapFromEmbeddedContext({ bootstrapMode: "full" })).toBe(true);
    expect(shouldStripBootstrapFromEmbeddedContext({ bootstrapMode: "limited" })).toBe(true);
    expect(shouldStripBootstrapFromEmbeddedContext({ bootstrapMode: "none" })).toBe(true);
  });
});

describe("isPrimaryBootstrapRun", () => {
  it("treats regular sessions as primary bootstrap runs", () => {
    expect(isPrimaryBootstrapRun("agent:main:main")).toBe(true);
  });

  it("suppresses bootstrap ownership for subagent and ACP/helper sessions", () => {
    expect(isPrimaryBootstrapRun("agent:main:subagent:worker")).toBe(false);
    expect(isPrimaryBootstrapRun("agent:main:acp:worker")).toBe(false);
  });
});

describe("remapInjectedContextFilesToWorkspace", () => {
  it("rewrites injected file paths onto the effective workspace when the tool root changes", () => {
    expect(
      remapInjectedContextFilesToWorkspace({
        files: [
          {
            path: "/real/workspace/AGENTS.md",
            content: "agents",
          },
          {
            path: "/real/workspace/nested/TOOLS.md",
            content: "tools",
          },
          {
            path: "/outside/README.md",
            content: "outside",
          },
        ],
        sourceWorkspaceDir: "/real/workspace",
        targetWorkspaceDir: "/sandbox/workspace",
      }),
    ).toEqual([
      {
        path: "/sandbox/workspace/AGENTS.md",
        content: "agents",
      },
      {
        path: "/sandbox/workspace/nested/TOOLS.md",
        content: "tools",
      },
      {
        path: "/outside/README.md",
        content: "outside",
      },
    ]);
  });
});

describe("shouldWarnOnOrphanedUserRepair", () => {
  it("warns for user and manual runs", () => {
    expect(shouldWarnOnOrphanedUserRepair("user")).toBe(true);
    expect(shouldWarnOnOrphanedUserRepair("manual")).toBe(true);
  });

  it("does not warn for background triggers", () => {
    expect(shouldWarnOnOrphanedUserRepair("heartbeat")).toBe(false);
    expect(shouldWarnOnOrphanedUserRepair("cron")).toBe(false);
    expect(shouldWarnOnOrphanedUserRepair("memory")).toBe(false);
    expect(shouldWarnOnOrphanedUserRepair("overflow")).toBe(false);
  });
});

describe("mergeOrphanedTrailingUserPrompt", () => {
  it("merges an orphaned user leaf into the next user-triggered prompt when missing", () => {
    expect(
      mergeOrphanedTrailingUserPrompt({
        prompt: "newest inbound message",
        trigger: "user",
        leafMessage: {
          content: [{ type: "text", text: "older active-turn message" }],
        } as never,
      }),
    ).toEqual({
      merged: true,
      removeLeaf: true,
      prompt:
        "[Queued user message that arrived while the previous turn was still active]\n" +
        "older active-turn message\n\nnewest inbound message",
    });
  });

  it("does not duplicate orphaned user text already present in the next prompt", () => {
    expect(
      mergeOrphanedTrailingUserPrompt({
        prompt: "summary\nolder active-turn message\nnewest inbound message",
        trigger: "user",
        leafMessage: {
          content: "older active-turn message",
        } as never,
      }),
    ).toEqual({
      merged: false,
      removeLeaf: true,
      prompt: "summary\nolder active-turn message\nnewest inbound message",
    });
  });

  it("does not treat short orphan text as duplicate from a substring match", () => {
    expect(
      mergeOrphanedTrailingUserPrompt({
        prompt: "please inspect this token",
        trigger: "user",
        leafMessage: {
          content: "ok",
        } as never,
      }),
    ).toEqual({
      merged: true,
      removeLeaf: true,
      prompt:
        "[Queued user message that arrived while the previous turn was still active]\n" +
        "ok\n\nplease inspect this token",
    });
  });

  it("preserves structured orphaned user content before removing the leaf", () => {
    expect(
      mergeOrphanedTrailingUserPrompt({
        prompt: "newest inbound message",
        trigger: "user",
        leafMessage: {
          content: [
            { type: "text", text: "please inspect this" },
            { type: "image_url", image_url: { url: "https://example.test/cat.png" } },
            { type: "input_audio", audio_url: "https://example.test/cat.wav" },
          ],
        } as never,
      }),
    ).toEqual({
      merged: true,
      removeLeaf: true,
      prompt:
        "[Queued user message that arrived while the previous turn was still active]\n" +
        "please inspect this\n" +
        "[image_url] https://example.test/cat.png\n" +
        "[input_audio] https://example.test/cat.wav\n\n" +
        "newest inbound message",
    });
  });

  it("summarizes inline structured media without embedding data URIs", () => {
    const dataUri = `data:image/png;base64,${"a".repeat(4096)}`;

    const result = mergeOrphanedTrailingUserPrompt({
      prompt: "newest inbound message",
      trigger: "user",
      leafMessage: {
        content: [
          { type: "text", text: "please inspect this inline image" },
          { type: "image_url", image_url: { url: dataUri } },
        ],
      } as never,
    });

    expect(result).toMatchObject({
      merged: true,
      removeLeaf: true,
    });
    expect(result.prompt).toContain("please inspect this inline image");
    expect(result.prompt).toContain("[image_url] inline data URI (image/png, 4118 chars)");
    expect(result.prompt).not.toContain("base64");
    expect(result.prompt).not.toContain("aaaa");
  });

  it("summarizes unknown structured data before JSON serialization", () => {
    const dataUri = `data:image/png;base64,${"a".repeat(10_000)}`;
    const result = mergeOrphanedTrailingUserPrompt({
      prompt: "newest inbound message",
      trigger: "user",
      leafMessage: {
        content: [
          {
            type: "unknown_content",
            nested: {
              inline: dataUri,
              longText: "b".repeat(2_000),
            },
          },
        ],
      } as never,
    });

    expect(result).toMatchObject({
      merged: true,
      removeLeaf: true,
    });
    expect(result.prompt).toContain("[value] inline data URI (image/png, 10022 chars)");
    expect(result.prompt).toContain("bbbb");
    expect(result.prompt).toContain("(2000 chars)");
    expect(result.prompt).not.toContain("base64");
    expect(result.prompt).not.toContain("aaaa");
  });

  it("removes an empty orphaned user leaf to prevent consecutive user turns", () => {
    expect(
      mergeOrphanedTrailingUserPrompt({
        prompt: "newest inbound message",
        trigger: "user",
        leafMessage: {
          content: [],
        } as never,
      }),
    ).toEqual({
      merged: false,
      removeLeaf: true,
      prompt: "newest inbound message",
    });
  });

  it("merges orphan prompt text for non-user triggers without warning policy changes", () => {
    expect(
      mergeOrphanedTrailingUserPrompt({
        prompt: "HEARTBEAT_OK",
        trigger: "heartbeat",
        leafMessage: {
          content: "older active-turn message",
        } as never,
      }),
    ).toEqual({
      merged: true,
      removeLeaf: true,
      prompt:
        "[Queued user message that arrived while the previous turn was still active]\n" +
        "older active-turn message\n\nHEARTBEAT_OK",
    });
  });
});

describe("resolveEmbeddedAgentStreamFn", () => {
  it("reuses the session's original base stream across later wrapper mutations", () => {
    resetEmbeddedAgentBaseStreamFnCacheForTest();
    const baseStreamFn = vi.fn();
    const wrapperStreamFn = vi.fn();
    const session = {
      agent: {
        streamFn: baseStreamFn,
      },
    };

    expect(resolveEmbeddedAgentBaseStreamFn({ session })).toBe(baseStreamFn);
    session.agent.streamFn = wrapperStreamFn;
    expect(resolveEmbeddedAgentBaseStreamFn({ session })).toBe(baseStreamFn);
  });

  it("injects authStorage api keys into provider-owned stream functions", async () => {
    const providerStreamFn = vi.fn(async (_model, _context, options) => options);
    const streamFn = resolveEmbeddedAgentStreamFn({
      currentStreamFn: undefined,
      providerStreamFn,
      shouldUseWebSocketTransport: false,
      sessionId: "session-1",
      model: {
        api: "openai-completions",
        provider: "demo-provider",
        id: "demo-model",
      } as never,
      authStorage: {
        getApiKey: vi.fn(async () => "demo-runtime-key"),
      },
    });

    await expect(
      streamFn({ provider: "demo-provider", id: "demo-model" } as never, {} as never, {}),
    ).resolves.toMatchObject({
      apiKey: "demo-runtime-key",
    });
    expect(providerStreamFn).toHaveBeenCalledTimes(1);
  });

  it("strips the internal cache boundary before provider-owned stream calls", async () => {
    const providerStreamFn = vi.fn(async (_model, context) => context);
    const streamFn = resolveEmbeddedAgentStreamFn({
      currentStreamFn: undefined,
      providerStreamFn,
      shouldUseWebSocketTransport: false,
      sessionId: "session-1",
      model: {
        api: "openai-completions",
        provider: "demo-provider",
        id: "demo-model",
      } as never,
    });

    await expect(
      streamFn(
        { provider: "demo-provider", id: "demo-model" } as never,
        {
          systemPrompt: `Stable prefix${SYSTEM_PROMPT_CACHE_BOUNDARY}Dynamic suffix`,
        } as never,
        {},
      ),
    ).resolves.toMatchObject({
      systemPrompt: "Stable prefix\nDynamic suffix",
    });
    expect(providerStreamFn).toHaveBeenCalledTimes(1);
  });
  it("routes supported default streamSimple fallbacks through boundary-aware transports", () => {
    const streamFn = resolveEmbeddedAgentStreamFn({
      currentStreamFn: undefined,
      shouldUseWebSocketTransport: false,
      sessionId: "session-1",
      model: {
        api: "openai-responses",
        provider: "openai",
        id: "gpt-5.4",
      } as never,
    });

    expect(streamFn).not.toBe(streamSimple);
  });

  it("keeps explicit custom currentStreamFn values unchanged", () => {
    const currentStreamFn = vi.fn();
    const streamFn = resolveEmbeddedAgentStreamFn({
      currentStreamFn: currentStreamFn as never,
      shouldUseWebSocketTransport: false,
      sessionId: "session-1",
      model: {
        api: "openai-responses",
        provider: "openai",
        id: "gpt-5.4",
      } as never,
    });

    expect(streamFn).toBe(currentStreamFn);
  });
});

describe("resolveAttemptFsWorkspaceOnly", () => {
  it("uses global tools.fs.workspaceOnly when agent has no override", () => {
    const cfg: OpenClawConfig = {
      tools: {
        fs: { workspaceOnly: true },
      },
    };

    expect(
      resolveAttemptFsWorkspaceOnly({
        config: cfg,
        sessionAgentId: "main",
      }),
    ).toBe(true);
  });

  it("prefers agent-specific tools.fs.workspaceOnly override", () => {
    const cfg: OpenClawConfig = {
      tools: {
        fs: { workspaceOnly: true },
      },
      agents: {
        list: [
          {
            id: "main",
            tools: {
              fs: { workspaceOnly: false },
            },
          },
        ],
      },
    };

    expect(
      resolveAttemptFsWorkspaceOnly({
        config: cfg,
        sessionAgentId: "main",
      }),
    ).toBe(false);
  });
});

describe("resolveUnknownToolGuardThreshold", () => {
  it("returns the default threshold when no loop-detection config is provided", () => {
    expect(resolveUnknownToolGuardThreshold(undefined)).toBe(10);
    expect(resolveUnknownToolGuardThreshold({})).toBe(10);
  });

  it("stays on even when tools.loopDetection.enabled is false (safety net)", () => {
    // The unknown-tool guard has no false-positive surface — the tool is
    // objectively not registered — so it is always on regardless of the
    // opt-in genericRepeat/pingPong/pollNoProgress detectors.
    expect(resolveUnknownToolGuardThreshold({ enabled: false })).toBe(10);
    expect(resolveUnknownToolGuardThreshold({ enabled: false, unknownToolThreshold: 3 })).toBe(3);
  });

  it("uses the configured threshold override when provided", () => {
    expect(resolveUnknownToolGuardThreshold({ enabled: true, unknownToolThreshold: 4 })).toBe(4);
  });

  it("falls back to the default threshold when the override is non-positive", () => {
    expect(resolveUnknownToolGuardThreshold({ unknownToolThreshold: 0 })).toBe(10);
    expect(resolveUnknownToolGuardThreshold({ unknownToolThreshold: -5 })).toBe(10);
    expect(resolveUnknownToolGuardThreshold({ unknownToolThreshold: Number.NaN })).toBe(10);
  });

  it("floors fractional overrides", () => {
    expect(resolveUnknownToolGuardThreshold({ unknownToolThreshold: 3.7 })).toBe(3);
  });
});

describe("wrapStreamFnTrimToolCallNames", () => {
  async function invokeWrappedStream(
    baseFn: (...args: never[]) => unknown,
    allowedToolNames?: Set<string>,
    guardOptions?: { unknownToolThreshold?: number },
  ) {
    return await invokeWrappedTestStream(
      (innerBaseFn) =>
        wrapStreamFnTrimToolCallNames(innerBaseFn as never, allowedToolNames, guardOptions),
      baseFn,
    );
  }

  function createEventStream(params: {
    event: unknown;
    finalToolCall: { type: string; name: string };
  }) {
    const finalMessage = { role: "assistant", content: [params.finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({ events: [params.event], resultMessage: finalMessage }),
    );
    return { baseFn, finalMessage };
  }

  it("trims whitespace from live streamed tool call names and final result message", async () => {
    const partialToolCall = { type: "toolCall", name: " read " };
    const messageToolCall = { type: "toolCall", name: " exec " };
    const finalToolCall = { type: "toolCall", name: " write " };
    const event = {
      type: "toolcall_delta",
      partial: { role: "assistant", content: [partialToolCall] },
      message: { role: "assistant", content: [messageToolCall] },
    };
    const { baseFn, finalMessage } = createEventStream({ event, finalToolCall });

    const stream = await invokeWrappedStream(baseFn);

    const seenEvents: unknown[] = [];
    for await (const item of stream) {
      seenEvents.push(item);
    }
    const result = await stream.result();

    expect(seenEvents).toHaveLength(1);
    expect(partialToolCall.name).toBe("read");
    expect(messageToolCall.name).toBe("exec");
    expect(finalToolCall.name).toBe("write");
    expect(result).toBe(finalMessage);
    expect(baseFn).toHaveBeenCalledTimes(1);
  });

  it("supports async stream functions that return a promise", async () => {
    const finalToolCall = { type: "toolCall", name: " browser " };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(async () =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn);
    const result = await stream.result();

    expect(finalToolCall.name).toBe("browser");
    expect(result).toBe(finalMessage);
    expect(baseFn).toHaveBeenCalledTimes(1);
  });
  it("normalizes common tool aliases when the canonical name is allowed", async () => {
    const finalToolCall = { type: "toolCall", name: " BASH " };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["exec"]));
    const result = await stream.result();

    expect(finalToolCall.name).toBe("exec");
    expect(result).toBe(finalMessage);
  });

  it("maps provider-prefixed tool names to allowed canonical tools", async () => {
    const partialToolCall = { type: "toolCall", name: " functions.read " };
    const messageToolCall = { type: "toolCall", name: " functions.write " };
    const finalToolCall = { type: "toolCall", name: " tools/exec " };
    const event = {
      type: "toolcall_delta",
      partial: { role: "assistant", content: [partialToolCall] },
      message: { role: "assistant", content: [messageToolCall] },
    };
    const { baseFn } = createEventStream({ event, finalToolCall });

    const stream = await invokeWrappedStream(baseFn, new Set(["read", "write", "exec"]));

    for await (const _item of stream) {
      // drain
    }
    await stream.result();

    expect(partialToolCall.name).toBe("read");
    expect(messageToolCall.name).toBe("write");
    expect(finalToolCall.name).toBe("exec");
  });

  it("normalizes toolUse and functionCall names before dispatch", async () => {
    const partialToolCall = { type: "toolUse", name: " functions.read " };
    const messageToolCall = { type: "functionCall", name: " functions.exec " };
    const finalToolCall = { type: "toolUse", name: " tools/write " };
    const event = {
      type: "toolcall_delta",
      partial: { role: "assistant", content: [partialToolCall] },
      message: { role: "assistant", content: [messageToolCall] },
    };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [event],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["read", "write", "exec"]));

    for await (const _item of stream) {
      // drain
    }
    const result = await stream.result();

    expect(partialToolCall.name).toBe("read");
    expect(messageToolCall.name).toBe("exec");
    expect(finalToolCall.name).toBe("write");
    expect(result).toBe(finalMessage);
  });

  it("preserves multi-segment tool suffixes when dropping provider prefixes", async () => {
    const finalToolCall = { type: "toolCall", name: " functions.graph.search " };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["graph.search", "search"]));
    const result = await stream.result();

    expect(finalToolCall.name).toBe("graph.search");
    expect(result).toBe(finalMessage);
  });

  it("rewrites repeated unavailable tool calls into plain assistant text after the threshold", async () => {
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: {
          role: "assistant",
          content: [{ type: "toolCall", name: " exec ", arguments: { command: "echo eleven" } }],
        },
      }),
    );
    const wrappedFn = wrapStreamFnTrimToolCallNames(baseFn as never, new Set(["read"]), {
      unknownToolThreshold: 10,
    });

    for (let i = 0; i < 10; i += 1) {
      const stream = await Promise.resolve(wrappedFn({} as never, {} as never, {} as never));
      const result = await stream.result();
      expect(result).toMatchObject({
        role: "assistant",
        content: [{ type: "toolCall", name: "exec" }],
      });
    }

    const blockedStream = await Promise.resolve(wrappedFn({} as never, {} as never, {} as never));
    const blockedResult = (await blockedStream.result()) as {
      role: string;
      content: Array<{ type: string; text?: string }>;
    };

    expect(blockedResult.role).toBe("assistant");
    expect(blockedResult.content).toEqual([
      expect.objectContaining({
        type: "text",
        text: expect.stringContaining('"exec"'),
      }),
    ]);
  });

  it("leaves repeated unavailable tool calls alone when the unknown-tool guard is disabled", async () => {
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: {
          role: "assistant",
          content: [{ type: "toolCall", name: " exec ", arguments: { command: "echo eleven" } }],
        },
      }),
    );
    const wrappedFn = wrapStreamFnTrimToolCallNames(baseFn as never, new Set(["read"]));

    for (let i = 0; i < 11; i += 1) {
      const stream = await Promise.resolve(wrappedFn({} as never, {} as never, {} as never));
      const result = await stream.result();
      expect(result).toMatchObject({
        role: "assistant",
        content: [{ type: "toolCall", name: "exec" }],
      });
    }
  });

  it("does not count partial tool-call deltas as separate unavailable-tool retries", async () => {
    const partialToolCall = { type: "toolCall", name: " exec " };
    const messageToolCall = { type: "toolCall", name: " exec " };
    const finalToolCall = { type: "toolCall", name: " exec " };
    const event = {
      type: "toolcall_delta",
      partial: { role: "assistant", content: [partialToolCall] },
      message: { role: "assistant", content: [messageToolCall] },
    };
    const { baseFn } = createEventStream({ event, finalToolCall });

    const stream = await invokeWrappedStream(baseFn, new Set(["read"]), {
      unknownToolThreshold: 1,
    });

    for await (const _item of stream) {
      // drain
    }
    const result = (await stream.result()) as {
      content: Array<{ type: string; text?: string; name?: string }>;
    };

    expect(partialToolCall.name).toBe("exec");
    expect(messageToolCall.name).toBe("exec");
    expect(result.content).toEqual([expect.objectContaining({ type: "toolCall", name: "exec" })]);
  });

  it("does not reset the unavailable-tool streak on partial-only stream chunks", async () => {
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [
          {
            type: "toolcall_delta",
            partial: { role: "assistant", content: [{ type: "toolCall", name: " exec " }] },
          },
        ],
        resultMessage: {
          role: "assistant",
          content: [{ type: "toolCall", name: " exec ", arguments: { command: "echo retry" } }],
        },
      }),
    );
    const wrappedFn = wrapStreamFnTrimToolCallNames(baseFn as never, new Set(["read"]), {
      unknownToolThreshold: 1,
    });

    const firstStream = await Promise.resolve(wrappedFn({} as never, {} as never, {} as never));
    await firstStream.result();

    const secondStream = await Promise.resolve(wrappedFn({} as never, {} as never, {} as never));
    for await (const _item of secondStream) {
      // drain
    }
    const secondResult = (await secondStream.result()) as {
      role: string;
      content: Array<{ type: string; text?: string; name?: string }>;
    };

    expect(secondResult.role).toBe("assistant");
    expect(secondResult.content).toEqual([
      expect.objectContaining({
        type: "text",
        text: expect.stringContaining('"exec"'),
      }),
    ]);
  });

  it("counts the final unknown-tool retry when streamed messages omit the tool name", async () => {
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [
          {
            type: "toolcall_delta",
            message: { role: "assistant", content: [{ type: "toolCall", name: "" }] },
          },
        ],
        resultMessage: {
          role: "assistant",
          content: [{ type: "toolCall", name: " exec ", arguments: { command: "echo retry" } }],
        },
      }),
    );
    const wrappedFn = wrapStreamFnTrimToolCallNames(baseFn as never, new Set(["read"]), {
      unknownToolThreshold: 1,
    });

    const firstStream = await Promise.resolve(wrappedFn({} as never, {} as never, {} as never));
    await firstStream.result();

    const secondStream = await Promise.resolve(wrappedFn({} as never, {} as never, {} as never));
    for await (const _item of secondStream) {
      // drain
    }
    const secondResult = (await secondStream.result()) as {
      role: string;
      content: Array<{ type: string; text?: string; name?: string }>;
    };

    expect(secondResult.role).toBe("assistant");
    expect(secondResult.content).toEqual([
      expect.objectContaining({
        type: "text",
        text: expect.stringContaining('"exec"'),
      }),
    ]);
  });

  it("resets a provisional streamed unknown-tool retry when later chunks resolve to an allowed tool", async () => {
    const baseFn = vi
      .fn()
      .mockImplementationOnce(() =>
        createFakeStream({
          events: [
            {
              type: "toolcall_delta",
              message: { role: "assistant", content: [{ type: "toolCall", name: " ex " }] },
            },
            {
              type: "toolcall_delta",
              message: { role: "assistant", content: [{ type: "toolCall", name: " exec " }] },
            },
          ],
          resultMessage: {
            role: "assistant",
            content: [{ type: "toolCall", name: " exec ", arguments: { command: "echo ok" } }],
          },
        }),
      )
      .mockImplementationOnce(() =>
        createFakeStream({
          events: [],
          resultMessage: {
            role: "assistant",
            content: [{ type: "toolCall", name: " ex ", arguments: { command: "echo retry" } }],
          },
        }),
      );
    const wrappedFn = wrapStreamFnTrimToolCallNames(baseFn as never, new Set(["exec"]), {
      unknownToolThreshold: 1,
    });

    const firstStream = await Promise.resolve(wrappedFn({} as never, {} as never, {} as never));
    for await (const _item of firstStream) {
      // drain
    }
    await firstStream.result();

    const secondStream = await Promise.resolve(wrappedFn({} as never, {} as never, {} as never));
    const secondResult = (await secondStream.result()) as {
      role: string;
      content: Array<{ type: string; text?: string; name?: string }>;
    };

    expect(secondResult.role).toBe("assistant");
    expect(secondResult.content).toEqual([
      expect.objectContaining({
        type: "toolCall",
        name: "ex",
      }),
    ]);
  });

  it("keeps processing later streamed messages after one streamed unknown-tool retry was counted", async () => {
    const baseFn = vi
      .fn()
      .mockImplementationOnce(() =>
        createFakeStream({
          events: [
            {
              type: "toolcall_delta",
              message: { role: "assistant", content: [{ type: "toolCall", name: " re " }] },
            },
            {
              type: "toolcall_delta",
              message: { role: "assistant", content: [{ type: "toolCall", name: " read " }] },
            },
          ],
          resultMessage: {
            role: "assistant",
            content: [{ type: "text", text: "resolved to allowed tool" }],
          },
        }),
      )
      .mockImplementationOnce(() =>
        createFakeStream({
          events: [],
          resultMessage: {
            role: "assistant",
            content: [{ type: "toolCall", name: " re ", arguments: { command: "echo retry" } }],
          },
        }),
      );
    const wrappedFn = wrapStreamFnTrimToolCallNames(baseFn as never, new Set(["read"]), {
      unknownToolThreshold: 1,
    });

    const firstStream = await Promise.resolve(wrappedFn({} as never, {} as never, {} as never));
    for await (const _item of firstStream) {
      // drain
    }
    await firstStream.result();

    const secondStream = await Promise.resolve(wrappedFn({} as never, {} as never, {} as never));
    const secondResult = (await secondStream.result()) as {
      role: string;
      content: Array<{ type: string; text?: string; name?: string }>;
    };

    expect(secondResult.role).toBe("assistant");
    expect(secondResult.content).toEqual([
      expect.objectContaining({
        type: "toolCall",
        name: "re",
      }),
    ]);
  });

  it("resets a stale unknown-tool streak when a streamed message mixes allowed and unknown tools", async () => {
    const baseFn = vi
      .fn()
      .mockImplementationOnce(() =>
        createFakeStream({
          events: [],
          resultMessage: {
            role: "assistant",
            content: [{ type: "toolCall", name: " ex ", arguments: { command: "echo first" } }],
          },
        }),
      )
      .mockImplementationOnce(() =>
        createFakeStream({
          events: [
            {
              type: "toolcall_delta",
              message: {
                role: "assistant",
                content: [
                  { type: "toolCall", name: " exec ", arguments: { command: "echo allowed" } },
                  { type: "toolCall", name: " ex ", arguments: { command: "echo provisional" } },
                ],
              },
            },
          ],
          resultMessage: {
            role: "assistant",
            content: [{ type: "toolCall", name: " exec ", arguments: { command: "echo ok" } }],
          },
        }),
      )
      .mockImplementationOnce(() =>
        createFakeStream({
          events: [],
          resultMessage: {
            role: "assistant",
            content: [{ type: "toolCall", name: " ex ", arguments: { command: "echo retry" } }],
          },
        }),
      );
    const wrappedFn = wrapStreamFnTrimToolCallNames(baseFn as never, new Set(["exec"]), {
      unknownToolThreshold: 1,
    });

    const firstStream = await Promise.resolve(wrappedFn({} as never, {} as never, {} as never));
    await firstStream.result();

    const secondStream = await Promise.resolve(wrappedFn({} as never, {} as never, {} as never));
    for await (const _item of secondStream) {
      // drain
    }
    await secondStream.result();

    const thirdStream = await Promise.resolve(wrappedFn({} as never, {} as never, {} as never));
    const thirdResult = (await thirdStream.result()) as {
      role: string;
      content: Array<{ type: string; text?: string; name?: string }>;
    };

    expect(thirdResult.role).toBe("assistant");
    expect(thirdResult.content).toEqual([
      expect.objectContaining({
        type: "toolCall",
        name: "ex",
      }),
    ]);
  });

  it("infers tool names from malformed toolCallId variants when allowlist is present", async () => {
    const partialToolCall = { type: "toolCall", id: "functions.read:0", name: "" };
    const finalToolCallA = { type: "toolCall", id: "functionsread3", name: "" };
    const finalToolCallB: { type: string; id: string; name?: string } = {
      type: "toolCall",
      id: "functionswrite4",
    };
    const finalToolCallC = { type: "functionCall", id: "functions.exec2", name: "" };
    const event = {
      type: "toolcall_delta",
      partial: { role: "assistant", content: [partialToolCall] },
    };
    const finalMessage = {
      role: "assistant",
      content: [finalToolCallA, finalToolCallB, finalToolCallC],
    };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [event],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["read", "write", "exec"]));
    for await (const _item of stream) {
      // drain
    }
    const result = await stream.result();

    expect(partialToolCall.name).toBe("read");
    expect(finalToolCallA.name).toBe("read");
    expect(finalToolCallB.name).toBe("write");
    expect(finalToolCallC.name).toBe("exec");
    expect(result).toBe(finalMessage);
  });

  it("does not infer names from malformed toolCallId when allowlist is absent", async () => {
    const finalToolCall: { type: string; id: string; name?: string } = {
      type: "toolCall",
      id: "functionsread3",
    };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn);
    await stream.result();

    expect(finalToolCall.name).toBeUndefined();
  });

  it("infers malformed non-blank tool names before dispatch", async () => {
    const partialToolCall = { type: "toolCall", id: "functionsread3", name: "functionsread3" };
    const finalToolCall = { type: "toolCall", id: "functionsread3", name: "functionsread3" };
    const event = {
      type: "toolcall_delta",
      partial: { role: "assistant", content: [partialToolCall] },
    };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [event],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["read", "write"]));
    for await (const _item of stream) {
      // drain
    }
    await stream.result();

    expect(partialToolCall.name).toBe("read");
    expect(finalToolCall.name).toBe("read");
  });

  it("recovers malformed non-blank names when id is missing", async () => {
    const finalToolCall = { type: "toolCall", name: "functionsread3" };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["read", "write"]));
    await stream.result();

    expect(finalToolCall.name).toBe("read");
  });

  it("recovers canonical tool names from canonical ids when name is empty", async () => {
    const finalToolCall = { type: "toolCall", id: "read", name: "" };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["read", "write"]));
    await stream.result();

    expect(finalToolCall.name).toBe("read");
  });

  it("recovers tool names from ids when name is whitespace-only", async () => {
    const finalToolCall = { type: "toolCall", id: "functionswrite4", name: "   " };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["read", "write"]));
    await stream.result();

    expect(finalToolCall.name).toBe("write");
  });

  it("keeps blank names blank and assigns fallback ids when both name and id are blank", async () => {
    const finalToolCall = { type: "toolCall", id: "", name: "" };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["read", "write"]));
    await stream.result();

    expect(finalToolCall.name).toBe("");
    expect(finalToolCall.id).toBe("call_auto_1");
  });

  it("assigns fallback ids when both name and id are missing", async () => {
    const finalToolCall: { type: string; name?: string; id?: string } = { type: "toolCall" };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["read", "write"]));
    await stream.result();

    expect(finalToolCall.name).toBeUndefined();
    expect(finalToolCall.id).toBe("call_auto_1");
  });

  it("prefers explicit canonical names over conflicting canonical ids", async () => {
    const finalToolCall = { type: "toolCall", id: "write", name: "read" };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["read", "write"]));
    await stream.result();

    expect(finalToolCall.name).toBe("read");
    expect(finalToolCall.id).toBe("write");
  });

  it("prefers explicit trimmed canonical names over conflicting malformed ids", async () => {
    const finalToolCall = { type: "toolCall", id: "functionswrite4", name: " read " };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["read", "write"]));
    await stream.result();

    expect(finalToolCall.name).toBe("read");
  });

  it("does not rewrite composite names that mention multiple tools", async () => {
    const finalToolCall = { type: "toolCall", id: "functionsread3", name: "read write" };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["read", "write"]));
    await stream.result();

    expect(finalToolCall.name).toBe("read write");
  });

  it("fails closed for malformed non-blank names that are ambiguous", async () => {
    const finalToolCall = { type: "toolCall", id: "functions.exec2", name: "functions.exec2" };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["exec", "exec2"]));
    await stream.result();

    expect(finalToolCall.name).toBe("functions.exec2");
  });

  it("matches malformed ids case-insensitively across common separators", async () => {
    const finalToolCall = { type: "toolCall", id: "Functions.Read_7", name: "" };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["read", "write"]));
    await stream.result();

    expect(finalToolCall.name).toBe("read");
  });
  it("does not override explicit non-blank tool names with inferred ids", async () => {
    const finalToolCall = { type: "toolCall", id: "functionswrite4", name: "someOtherTool" };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["read", "write"]));
    await stream.result();

    expect(finalToolCall.name).toBe("someOtherTool");
  });

  it("fails closed when malformed ids could map to multiple allowlisted tools", async () => {
    const finalToolCall = { type: "toolCall", id: "functions.exec2", name: "" };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn, new Set(["exec", "exec2"]));
    await stream.result();

    expect(finalToolCall.name).toBe("");
  });
  it("does not collapse whitespace-only tool names to empty strings", async () => {
    const partialToolCall = { type: "toolCall", name: "   " };
    const finalToolCall = { type: "toolCall", name: "\t  " };
    const event = {
      type: "toolcall_delta",
      partial: { role: "assistant", content: [partialToolCall] },
    };
    const { baseFn } = createEventStream({ event, finalToolCall });

    const stream = await invokeWrappedStream(baseFn);

    for await (const _item of stream) {
      // drain
    }
    await stream.result();

    expect(partialToolCall.name).toBe("   ");
    expect(finalToolCall.name).toBe("\t  ");
    expect(baseFn).toHaveBeenCalledTimes(1);
  });

  it("assigns fallback ids to missing/blank tool call ids in streamed and final messages", async () => {
    const partialToolCall = { type: "toolCall", name: " read ", id: "   " };
    const finalToolCallA = { type: "toolCall", name: " exec ", id: "" };
    const finalToolCallB: { type: string; name: string; id?: string } = {
      type: "toolCall",
      name: " write ",
    };
    const event = {
      type: "toolcall_delta",
      partial: { role: "assistant", content: [partialToolCall] },
    };
    const finalMessage = { role: "assistant", content: [finalToolCallA, finalToolCallB] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [event],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn);
    for await (const _item of stream) {
      // drain
    }
    const result = await stream.result();

    expect(partialToolCall.name).toBe("read");
    expect(partialToolCall.id).toBe("call_auto_1");
    expect(finalToolCallA.name).toBe("exec");
    expect(finalToolCallA.id).toBe("call_auto_1");
    expect(finalToolCallB.name).toBe("write");
    expect(finalToolCallB.id).toBe("call_auto_2");
    expect(result).toBe(finalMessage);
  });

  it("trims surrounding whitespace on tool call ids", async () => {
    const finalToolCall = { type: "toolCall", name: " read ", id: "  call_42  " };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn);
    await stream.result();

    expect(finalToolCall.name).toBe("read");
    expect(finalToolCall.id).toBe("call_42");
  });

  it("reassigns duplicate tool call ids within a message to unique fallbacks", async () => {
    const finalToolCallA = { type: "toolCall", name: " read ", id: "  edit:22  " };
    const finalToolCallB = { type: "toolCall", name: " write ", id: "edit:22" };
    const finalMessage = { role: "assistant", content: [finalToolCallA, finalToolCallB] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn);
    await stream.result();

    expect(finalToolCallA.name).toBe("read");
    expect(finalToolCallB.name).toBe("write");
    expect(finalToolCallA.id).toBe("edit:22");
    expect(finalToolCallB.id).toBe("call_auto_1");
  });
});

describe("wrapStreamFnSanitizeMalformedToolCalls", () => {
  it("drops malformed assistant tool calls from outbound context before provider replay", async () => {
    const messages = [
      {
        role: "assistant",
        stopReason: "error",
        content: [{ type: "toolCall", name: "read", arguments: {} }],
      },
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]), {
      validateAnthropicTurns: true,
      preserveSignatures: true,
      dropThinkingBlocks: false,
    } as never);
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as { messages: unknown[] };
    expect(seenContext.messages).toEqual([
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ]);
    expect(seenContext.messages).not.toBe(messages);
  });

  it("preserves outbound context when all assistant tool calls are valid", async () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "call_1", name: "read", arguments: {} }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]), {
      validateAnthropicTurns: true,
      preserveSignatures: true,
      dropThinkingBlocks: false,
    } as never);
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as { messages: unknown[] };
    expect(seenContext.messages).toBe(messages);
  });

  it("strips trailing assistant prefill turns for Anthropic outbound replay", async () => {
    const messages = [
      {
        role: "user",
        content: [{ type: "text", text: "earlier question" }],
      },
      {
        role: "assistant",
        content: [{ type: "text", text: "stale assistant answer" }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]), {
      validateAnthropicTurns: true,
      preserveSignatures: true,
      dropThinkingBlocks: false,
    } as never);
    const stream = wrapped(
      { api: "anthropic-messages" } as never,
      { messages } as never,
      {} as never,
    ) as FakeWrappedStream | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as { messages: unknown[] };
    expect(seenContext.messages).toEqual([
      {
        role: "user",
        content: [{ type: "text", text: "earlier question" }],
      },
    ]);
    expect(seenContext.messages).not.toBe(messages);
  });

  it("strips trailing assistant prefill turns for Gemini outbound replay", async () => {
    const messages = [
      {
        role: "user",
        content: [{ type: "text", text: "earlier question" }],
      },
      {
        role: "assistant",
        content: [{ type: "text", text: "stale model answer" }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]), {
      validateGeminiTurns: true,
      preserveSignatures: true,
      dropThinkingBlocks: false,
    } as never);
    const stream = wrapped(
      { api: "google-generative-ai" } as never,
      { messages } as never,
      {} as never,
    ) as FakeWrappedStream | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as { messages: unknown[] };
    expect(seenContext.messages).toEqual([
      {
        role: "user",
        content: [{ type: "text", text: "earlier question" }],
      },
    ]);
    expect(seenContext.messages).not.toBe(messages);
  });

  it("drops signed thinking turns when sibling replay tool calls are not allowlisted", async () => {
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "internal", thinkingSignature: "sig_1" },
          { type: "toolCall", id: "toolu_legacy", name: "gateway", arguments: {} },
        ],
      },
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]), {
      validateAnthropicTurns: true,
      preserveSignatures: true,
      dropThinkingBlocks: false,
    } as never);
    const stream = wrapped(
      { api: "anthropic-messages" } as never,
      { messages } as never,
      {} as never,
    ) as FakeWrappedStream | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as { messages: unknown[] };
    expect(seenContext.messages).toEqual([
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ]);
  });

  it("drops signed thinking turns for bedrock claude replay when sibling tool calls are not replay-safe", async () => {
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "internal", thinkingSignature: "sig_1" },
          { type: "toolCall", id: "toolu_legacy", name: "gateway", arguments: {} },
        ],
      },
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]), {
      validateAnthropicTurns: true,
      preserveSignatures: true,
      dropThinkingBlocks: false,
    } as never);
    const stream = wrapped(
      { api: "bedrock-converse-stream" } as never,
      { messages } as never,
      {} as never,
    ) as FakeWrappedStream | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as { messages: unknown[] };
    expect(seenContext.messages).toEqual([
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ]);
  });

  it("drops signed thinking turns when sibling replay tool calls reuse an id", async () => {
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "internal", thinkingSignature: "sig_1" },
          { type: "toolCall", id: "call_1", name: "read", arguments: {} },
          { type: "functionCall", id: "call_1", name: "read", arguments: {} },
        ],
      },
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]), {
      validateAnthropicTurns: true,
      preserveSignatures: true,
      dropThinkingBlocks: false,
    } as never);
    const stream = wrapped(
      { api: "anthropic-messages" } as never,
      { messages } as never,
      {} as never,
    ) as FakeWrappedStream | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as { messages: unknown[] };
    expect(seenContext.messages).toEqual([
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ]);
  });

  it("drops signed thinking turns when replay would expose inline sessions_spawn attachments", async () => {
    const attachmentContent = "SIGNED_THINKING_INLINE_ATTACHMENT";
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "internal", thinkingSignature: "sig_1" },
          {
            type: "toolUse",
            id: "call_1",
            name: "sessions_spawn",
            input: {
              task: "inspect attachment",
              attachments: [{ name: "snapshot.txt", content: attachmentContent }],
            },
          },
        ],
      },
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(
      baseFn as never,
      new Set(["sessions_spawn"]),
      {
        validateAnthropicTurns: true,
        preserveSignatures: true,
        dropThinkingBlocks: false,
      } as never,
    );
    const stream = wrapped(
      { api: "anthropic-messages" } as never,
      { messages } as never,
      {} as never,
    ) as FakeWrappedStream | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as { messages: unknown[] };
    expect(seenContext.messages).toEqual([
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ]);
  });

  it("drops signed thinking turns when replay would expose non-content attachment payload fields", async () => {
    const attachmentContent = "SIGNED_THINKING_NESTED_ATTACHMENT";
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "internal", thinkingSignature: "sig_1" },
          {
            type: "toolUse",
            id: "call_1",
            name: "sessions_spawn",
            input: {
              task: "inspect attachment",
              attachments: [
                {
                  name: "snapshot.txt",
                  mimeType: "text/plain",
                  data: attachmentContent,
                },
              ],
            },
          },
        ],
      },
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(
      baseFn as never,
      new Set(["sessions_spawn"]),
      {
        validateAnthropicTurns: true,
        preserveSignatures: true,
        dropThinkingBlocks: false,
      } as never,
    );
    const stream = wrapped(
      { api: "anthropic-messages" } as never,
      { messages } as never,
      {} as never,
    ) as FakeWrappedStream | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as { messages: unknown[] };
    expect(seenContext.messages).toEqual([
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ]);
  });

  it("keeps mutable thinking turns outside anthropic replay-only preservation", async () => {
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "internal", thinkingSignature: "sig_1" },
          { type: "toolCall", id: "call_1", name: " read ", arguments: {} },
        ],
      },
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]), {
      validateAnthropicTurns: true,
    } as never);
    const stream = wrapped(
      { api: "openai-completions" } as never,
      { messages } as never,
      {} as never,
    ) as FakeWrappedStream | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as { messages: unknown[] };
    expect(seenContext.messages).toEqual([
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "internal", thinkingSignature: "sig_1" },
          { type: "toolCall", id: "call_1", name: "read", arguments: {} },
        ],
      },
      {
        role: "toolResult",
        toolCallId: "call_1",
        toolName: "read",
        content: [
          {
            type: "text",
            text: "[openclaw] missing tool result in session history; inserted synthetic error result for transcript repair.",
          },
        ],
        isError: true,
        timestamp: expect.any(Number),
      },
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ]);
  });

  it("preserves sessions_spawn attachment payloads on replay", async () => {
    const attachmentContent = "INLINE_ATTACHMENT_PAYLOAD";
    const messages = [
      {
        role: "assistant",
        content: [
          {
            type: "toolUse",
            id: "call_1",
            name: "  SESSIONS_SPAWN  ",
            input: {
              task: "inspect attachment",
              attachments: [{ name: "snapshot.txt", content: attachmentContent }],
            },
          },
        ],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(
      baseFn as never,
      new Set(["sessions_spawn"]),
      { validateAnthropicTurns: true } as never,
    );
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as {
      messages: Array<{ content?: Array<Record<string, unknown>> }>;
    };
    const toolCall = seenContext.messages[0]?.content?.[0] as {
      name?: string;
      input?: { attachments?: Array<{ content?: string }> };
    };
    expect(toolCall.name).toBe("sessions_spawn");
    expect(toolCall.input?.attachments?.[0]?.content).toBe(attachmentContent);
  });

  it("keeps non-Anthropic thinking turns mutable when Anthropic replay validation is off", async () => {
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "internal", thinkingSignature: "sig_1" },
          { type: "toolCall", id: "call_read", name: " read ", arguments: { path: "README.md" } },
        ],
      },
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]));
    const stream = wrapped(
      { api: "google-gemini" } as never,
      { messages } as never,
      {} as never,
    ) as FakeWrappedStream | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as {
      messages: Array<{ content?: unknown[] }>;
    };
    expect(seenContext.messages[0]?.content).toEqual([
      { type: "thinking", thinking: "internal", thinkingSignature: "sig_1" },
      { type: "toolCall", id: "call_read", name: "read", arguments: { path: "README.md" } },
    ]);
  });

  it("preserves allowlisted tool names that contain punctuation", async () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "toolUse", id: "call_1", name: "admin.export", input: { scope: "all" } }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(
      baseFn as never,
      new Set(["admin.export"]),
    );
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as { messages: unknown[] };
    expect(seenContext.messages).toBe(messages);
  });

  it("normalizes provider-prefixed replayed tool names before provider replay", async () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "toolUse", id: "call_1", name: "functions.read", input: { path: "." } }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]));
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as {
      messages: Array<{ content?: Array<{ name?: string }> }>;
    };
    expect(seenContext.messages[0]?.content?.[0]?.name).toBe("read");
  });

  it("canonicalizes mixed-case allowlisted tool names on replay", async () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "call_1", name: "readfile", arguments: {} }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["ReadFile"]));
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as {
      messages: Array<{ content?: Array<{ name?: string }> }>;
    };
    expect(seenContext.messages[0]?.content?.[0]?.name).toBe("ReadFile");
  });

  it("recovers blank replayed tool names from their ids", async () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "functionswrite4", name: "   ", arguments: {} }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["write"]));
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as {
      messages: Array<{ content?: Array<{ name?: string }> }>;
    };
    expect(seenContext.messages[0]?.content?.[0]?.name).toBe("write");
  });

  it("recovers mangled replayed tool names before dropping the call", async () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "call_1", name: "functionsread3", arguments: {} }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]));
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as {
      messages: Array<{ content?: Array<{ name?: string }> }>;
    };
    expect(seenContext.messages[0]?.content?.[0]?.name).toBe("read");
  });

  it("drops orphaned tool results after replay sanitization removes a tool-call turn", async () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "toolCall", name: "read", arguments: {} }],
        stopReason: "error",
      },
      {
        role: "toolResult",
        toolCallId: "call_missing",
        toolName: "read",
        content: [{ type: "text", text: "stale result" }],
        isError: false,
      },
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]));
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as {
      messages: Array<{ role?: string }>;
    };
    expect(seenContext.messages).toEqual([
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ]);
  });

  it("drops replayed tool calls that are no longer allowlisted", async () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "call_1", name: "write", arguments: {} }],
      },
      {
        role: "toolResult",
        toolCallId: "call_1",
        toolName: "write",
        content: [{ type: "text", text: "stale result" }],
        isError: false,
      },
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]));
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as {
      messages: Array<{ role?: string }>;
    };
    expect(seenContext.messages).toEqual([
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ]);
  });
  it("drops replayed tool names that are no longer allowlisted", async () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "toolUse", id: "call_1", name: "unknown_tool", input: { path: "." } }],
      },
      {
        role: "toolResult",
        toolCallId: "call_1",
        toolName: "unknown_tool",
        content: [{ type: "text", text: "stale result" }],
        isError: false,
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]));
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as { messages: unknown[] };
    expect(seenContext.messages).toEqual([]);
  });

  it("drops ambiguous mangled replay names instead of guessing a tool", async () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "call_1", name: "functions.exec2", arguments: {} }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(
      baseFn as never,
      new Set(["exec", "exec2"]),
    );
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as { messages: unknown[] };
    expect(seenContext.messages).toEqual([]);
  });

  it("preserves matching tool results for retained errored assistant turns", async () => {
    const messages = [
      {
        role: "assistant",
        stopReason: "error",
        content: [
          { type: "toolCall", id: "call_1", name: "read", arguments: {} },
          { type: "toolCall", name: "read", arguments: {} },
        ],
      },
      {
        role: "toolResult",
        toolCallId: "call_1",
        toolName: "read",
        content: [{ type: "text", text: "kept result" }],
        isError: false,
      },
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]));
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as { messages: unknown[] };
    expect(seenContext.messages).toEqual([
      {
        role: "assistant",
        stopReason: "error",
        content: [{ type: "toolCall", id: "call_1", name: "read", arguments: {} }],
      },
      {
        role: "toolResult",
        toolCallId: "call_1",
        toolName: "read",
        content: [{ type: "text", text: "kept result" }],
        isError: false,
      },
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ]);
  });

  it("revalidates turn ordering after dropping an assistant replay turn", async () => {
    const messages = [
      {
        role: "user",
        content: [{ type: "text", text: "first" }],
      },
      {
        role: "assistant",
        stopReason: "error",
        content: [{ type: "toolCall", name: "read", arguments: {} }],
      },
      {
        role: "user",
        content: [{ type: "text", text: "second" }],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]), {
      validateGeminiTurns: false,
      validateAnthropicTurns: true,
      preserveSignatures: false,
      dropThinkingBlocks: false,
    });
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as {
      messages: Array<{ role?: string; content?: unknown[] }>;
    };
    expect(seenContext.messages).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "first" },
          { type: "text", text: "second" },
        ],
      },
    ]);
  });

  it("drops orphaned Anthropic user tool_result blocks after replay sanitization", async () => {
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "partial response" },
          { type: "toolUse", name: "read", input: { path: "." } },
        ],
      },
      {
        role: "user",
        content: [
          { type: "toolResult", toolUseId: "call_1", content: [{ type: "text", text: "stale" }] },
          { type: "text", text: "retry" },
        ],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]), {
      validateGeminiTurns: false,
      validateAnthropicTurns: true,
      preserveSignatures: false,
      dropThinkingBlocks: false,
    });
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as {
      messages: Array<{ role?: string; content?: unknown[] }>;
    };
    expect(seenContext.messages).toEqual([
      {
        role: "assistant",
        content: [{ type: "text", text: "partial response" }],
      },
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ]);
  });

  it("drops embedded Anthropic user tool_result blocks when signed-thinking replay must stay provider-owned", async () => {
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "internal", thinkingSignature: "sig_1" },
          { type: "toolUse", id: "call_1", name: "read", input: { path: "." } },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "toolResult",
            toolUseId: "call_1",
            content: [{ type: "text", text: "embedded result" }],
          },
          { type: "text", text: "retry" },
        ],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]), {
      validateGeminiTurns: false,
      validateAnthropicTurns: true,
      preserveSignatures: true,
      dropThinkingBlocks: false,
    });
    const stream = wrapped(
      { api: "anthropic-messages" } as never,
      { messages } as never,
      {} as never,
    ) as FakeWrappedStream | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as {
      messages: Array<{ role?: string; content?: unknown[] }>;
    };
    expect(seenContext.messages).toEqual([
      {
        role: "assistant",
        content: [{ type: "text", text: "[tool calls omitted]" }],
      },
      {
        role: "user",
        content: [{ type: "text", text: "retry" }],
      },
    ]);
  });

  it("preserves embedded Anthropic user tool_result blocks for non-thinking turns even when immutable replay is enabled", async () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "toolUse", id: "call_1", name: "read", input: { path: "." } }],
      },
      {
        role: "user",
        content: [
          {
            type: "toolResult",
            toolUseId: "call_1",
            content: [{ type: "text", text: "kept result" }],
          },
          { type: "text", text: "retry" },
        ],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]), {
      validateGeminiTurns: false,
      validateAnthropicTurns: true,
      preserveSignatures: true,
      dropThinkingBlocks: false,
    });
    const stream = wrapped(
      { api: "anthropic-messages" } as never,
      { messages } as never,
      {} as never,
    ) as FakeWrappedStream | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as {
      messages: Array<{ role?: string; content?: unknown[] }>;
    };
    expect(seenContext.messages).toEqual(messages);
  });

  it.each(["toolCall", "functionCall"] as const)(
    "preserves matching Anthropic user tool_result blocks after %s replay turns",
    async (toolCallType) => {
      const messages = [
        {
          role: "assistant",
          content: [{ type: toolCallType, id: "call_1", name: "read", arguments: {} }],
        },
        {
          role: "user",
          content: [
            {
              type: "toolResult",
              toolUseId: "call_1",
              content: [{ type: "text", text: "kept result" }],
            },
            { type: "text", text: "retry" },
          ],
        },
      ];
      const baseFn = vi.fn((_model, _context) =>
        createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
      );

      const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]), {
        validateGeminiTurns: false,
        validateAnthropicTurns: true,
        preserveSignatures: false,
        dropThinkingBlocks: false,
      });
      const stream = wrapped({} as never, { messages } as never, {} as never) as
        | FakeWrappedStream
        | Promise<FakeWrappedStream>;
      await Promise.resolve(stream);

      expect(baseFn).toHaveBeenCalledTimes(1);
      const seenContext = baseFn.mock.calls[0]?.[1] as {
        messages: Array<{ role?: string; content?: unknown[] }>;
      };
      expect(seenContext.messages).toEqual(messages);
    },
  );

  it("drops orphaned Anthropic user tool_result blocks after dropping an assistant replay turn", async () => {
    const messages = [
      {
        role: "user",
        content: [{ type: "text", text: "first" }],
      },
      {
        role: "assistant",
        stopReason: "error",
        content: [{ type: "toolUse", name: "read", input: { path: "." } }],
      },
      {
        role: "user",
        content: [
          { type: "toolResult", toolUseId: "call_1", content: [{ type: "text", text: "stale" }] },
          { type: "text", text: "second" },
        ],
      },
    ];
    const baseFn = vi.fn((_model, _context) =>
      createFakeStream({ events: [], resultMessage: { role: "assistant", content: [] } }),
    );

    const wrapped = wrapStreamFnSanitizeMalformedToolCalls(baseFn as never, new Set(["read"]), {
      validateGeminiTurns: false,
      validateAnthropicTurns: true,
      preserveSignatures: false,
      dropThinkingBlocks: false,
    });
    const stream = wrapped({} as never, { messages } as never, {} as never) as
      | FakeWrappedStream
      | Promise<FakeWrappedStream>;
    await Promise.resolve(stream);

    expect(baseFn).toHaveBeenCalledTimes(1);
    const seenContext = baseFn.mock.calls[0]?.[1] as {
      messages: Array<{ role?: string; content?: unknown[] }>;
    };
    expect(seenContext.messages).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "first" },
          { type: "text", text: "second" },
        ],
      },
    ]);
  });
});

describe("wrapStreamFnRepairMalformedToolCallArguments", () => {
  async function invokeWrappedStream(baseFn: (...args: never[]) => unknown) {
    return await invokeWrappedTestStream(
      (innerBaseFn) => wrapStreamFnRepairMalformedToolCallArguments(innerBaseFn as never),
      baseFn,
    );
  }

  it("repairs anthropic-compatible tool arguments when trailing junk follows valid JSON", async () => {
    const partialToolCall = { type: "toolCall", name: "read", arguments: {} };
    const streamedToolCall = { type: "toolCall", name: "read", arguments: {} };
    const endMessageToolCall = { type: "toolCall", name: "read", arguments: {} };
    const finalToolCall = { type: "toolCall", name: "read", arguments: {} };
    const partialMessage = { role: "assistant", content: [partialToolCall] };
    const endMessage = { role: "assistant", content: [endMessageToolCall] };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [
          {
            type: "toolcall_delta",
            contentIndex: 0,
            delta: '{"path":"/tmp/report.txt"}',
            partial: partialMessage,
          },
          {
            type: "toolcall_delta",
            contentIndex: 0,
            delta: "xx",
            partial: partialMessage,
          },
          {
            type: "toolcall_end",
            contentIndex: 0,
            toolCall: streamedToolCall,
            partial: partialMessage,
            message: endMessage,
          },
        ],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn);
    for await (const _item of stream) {
      // drain
    }
    const result = await stream.result();

    expect(partialToolCall.arguments).toEqual({ path: "/tmp/report.txt" });
    expect(streamedToolCall.arguments).toEqual({ path: "/tmp/report.txt" });
    expect(endMessageToolCall.arguments).toEqual({ path: "/tmp/report.txt" });
    expect(finalToolCall.arguments).toEqual({ path: "/tmp/report.txt" });
    expect(result).toBe(finalMessage);
  });

  it("repairs tool arguments when malformed tool-call preamble appears before JSON", async () => {
    const partialToolCall = { type: "toolCall", name: "write", arguments: {} };
    const streamedToolCall = { type: "toolCall", name: "write", arguments: {} };
    const endMessageToolCall = { type: "toolCall", name: "write", arguments: {} };
    const finalToolCall = { type: "toolCall", name: "write", arguments: {} };
    const partialMessage = { role: "assistant", content: [partialToolCall] };
    const endMessage = { role: "assistant", content: [endMessageToolCall] };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [
          {
            type: "toolcall_delta",
            contentIndex: 0,
            delta: '.functions.write:8  \n{"path":"/tmp/report.txt"}',
            partial: partialMessage,
          },
          {
            type: "toolcall_end",
            contentIndex: 0,
            toolCall: streamedToolCall,
            partial: partialMessage,
            message: endMessage,
          },
        ],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn);
    for await (const _item of stream) {
      // drain
    }
    const result = await stream.result();

    expect(partialToolCall.arguments).toEqual({ path: "/tmp/report.txt" });
    expect(streamedToolCall.arguments).toEqual({ path: "/tmp/report.txt" });
    expect(endMessageToolCall.arguments).toEqual({ path: "/tmp/report.txt" });
    expect(finalToolCall.arguments).toEqual({ path: "/tmp/report.txt" });
    expect(result).toBe(finalMessage);
  });
  it("preserves anthropic-compatible tool arguments when the streamed JSON is already valid", async () => {
    const partialToolCall = { type: "toolCall", name: "read", arguments: {} };
    const streamedToolCall = { type: "toolCall", name: "read", arguments: {} };
    const endMessageToolCall = { type: "toolCall", name: "read", arguments: {} };
    const finalToolCall = { type: "toolCall", name: "read", arguments: {} };
    const partialMessage = { role: "assistant", content: [partialToolCall] };
    const endMessage = { role: "assistant", content: [endMessageToolCall] };
    const finalMessage = { role: "assistant", content: [finalToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [
          {
            type: "toolcall_delta",
            contentIndex: 0,
            delta: '{"path":"/tmp/report.txt"',
            partial: partialMessage,
          },
          {
            type: "toolcall_delta",
            contentIndex: 0,
            delta: "}",
            partial: partialMessage,
          },
          {
            type: "toolcall_end",
            contentIndex: 0,
            toolCall: streamedToolCall,
            partial: partialMessage,
            message: endMessage,
          },
        ],
        resultMessage: finalMessage,
      }),
    );

    const stream = await invokeWrappedStream(baseFn);
    for await (const _item of stream) {
      // drain
    }
    const result = await stream.result();

    expect(partialToolCall.arguments).toEqual({ path: "/tmp/report.txt" });
    expect(streamedToolCall.arguments).toEqual({ path: "/tmp/report.txt" });
    expect(endMessageToolCall.arguments).toEqual({ path: "/tmp/report.txt" });
    expect(finalToolCall.arguments).toEqual({ path: "/tmp/report.txt" });
    expect(result).toBe(finalMessage);
  });

  it("does not repair tool arguments when leading text is not tool-call metadata", async () => {
    const partialToolCall = { type: "toolCall", name: "read", arguments: {} };
    const streamedToolCall = { type: "toolCall", name: "read", arguments: {} };
    const partialMessage = { role: "assistant", content: [partialToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [
          {
            type: "toolcall_delta",
            contentIndex: 0,
            delta: 'please use {"path":"/tmp/report.txt"}',
            partial: partialMessage,
          },
          {
            type: "toolcall_end",
            contentIndex: 0,
            toolCall: streamedToolCall,
            partial: partialMessage,
          },
        ],
        resultMessage: { role: "assistant", content: [partialToolCall] },
      }),
    );

    const stream = await invokeWrappedStream(baseFn);
    for await (const _item of stream) {
      // drain
    }

    expect(partialToolCall.arguments).toEqual({});
    expect(streamedToolCall.arguments).toEqual({});
  });

  it("keeps incomplete partial JSON unchanged until a complete object exists", async () => {
    const partialToolCall = { type: "toolCall", name: "read", arguments: {} };
    const partialMessage = { role: "assistant", content: [partialToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [
          {
            type: "toolcall_delta",
            contentIndex: 0,
            delta: '{"path":"/tmp',
            partial: partialMessage,
          },
        ],
        resultMessage: { role: "assistant", content: [partialToolCall] },
      }),
    );

    const stream = await invokeWrappedStream(baseFn);
    for await (const _item of stream) {
      // drain
    }

    expect(partialToolCall.arguments).toEqual({});
  });

  it("does not repair tool arguments when trailing junk exceeds the Kimi-specific allowance", async () => {
    const partialToolCall = { type: "toolCall", name: "read", arguments: {} };
    const streamedToolCall = { type: "toolCall", name: "read", arguments: {} };
    const partialMessage = { role: "assistant", content: [partialToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [
          {
            type: "toolcall_delta",
            contentIndex: 0,
            delta: '{"path":"/tmp/report.txt"}oops',
            partial: partialMessage,
          },
          {
            type: "toolcall_end",
            contentIndex: 0,
            toolCall: streamedToolCall,
            partial: partialMessage,
          },
        ],
        resultMessage: { role: "assistant", content: [partialToolCall] },
      }),
    );

    const stream = await invokeWrappedStream(baseFn);
    for await (const _item of stream) {
      // drain
    }

    expect(partialToolCall.arguments).toEqual({});
    expect(streamedToolCall.arguments).toEqual({});
  });

  it("clears a cached repair when later deltas make the trailing suffix invalid", async () => {
    const partialToolCall = { type: "toolCall", name: "read", arguments: {} };
    const streamedToolCall = { type: "toolCall", name: "read", arguments: {} };
    const partialMessage = { role: "assistant", content: [partialToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [
          {
            type: "toolcall_delta",
            contentIndex: 0,
            delta: '{"path":"/tmp/report.txt"}',
            partial: partialMessage,
          },
          {
            type: "toolcall_delta",
            contentIndex: 0,
            delta: "x",
            partial: partialMessage,
          },
          {
            type: "toolcall_delta",
            contentIndex: 0,
            delta: "yzq",
            partial: partialMessage,
          },
          {
            type: "toolcall_end",
            contentIndex: 0,
            toolCall: streamedToolCall,
            partial: partialMessage,
          },
        ],
        resultMessage: { role: "assistant", content: [partialToolCall] },
      }),
    );

    const stream = await invokeWrappedStream(baseFn);
    for await (const _item of stream) {
      // drain
    }

    expect(partialToolCall.arguments).toEqual({});
    expect(streamedToolCall.arguments).toEqual({});
  });

  it("clears a cached repair when a later delta adds a single oversized trailing suffix", async () => {
    const partialToolCall = { type: "toolCall", name: "read", arguments: {} };
    const streamedToolCall = { type: "toolCall", name: "read", arguments: {} };
    const partialMessage = { role: "assistant", content: [partialToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [
          {
            type: "toolcall_delta",
            contentIndex: 0,
            delta: '{"path":"/tmp/report.txt"}',
            partial: partialMessage,
          },
          {
            type: "toolcall_delta",
            contentIndex: 0,
            delta: "oops",
            partial: partialMessage,
          },
          {
            type: "toolcall_end",
            contentIndex: 0,
            toolCall: streamedToolCall,
            partial: partialMessage,
          },
        ],
        resultMessage: { role: "assistant", content: [partialToolCall] },
      }),
    );

    const stream = await invokeWrappedStream(baseFn);
    for await (const _item of stream) {
      // drain
    }

    expect(partialToolCall.arguments).toEqual({});
    expect(streamedToolCall.arguments).toEqual({});
  });

  it("preserves preexisting tool arguments when later reevaluation fails", async () => {
    const partialToolCall = {
      type: "toolCall",
      name: "read",
      arguments: { path: "/etc/hosts" },
    };
    const streamedToolCall = { type: "toolCall", name: "read", arguments: {} };
    const partialMessage = { role: "assistant", content: [partialToolCall] };
    const baseFn = vi.fn(() =>
      createFakeStream({
        events: [
          {
            type: "toolcall_delta",
            contentIndex: 0,
            delta: "}",
            partial: partialMessage,
          },
          {
            type: "toolcall_end",
            contentIndex: 0,
            toolCall: streamedToolCall,
            partial: partialMessage,
          },
        ],
        resultMessage: { role: "assistant", content: [partialToolCall] },
      }),
    );

    const stream = await invokeWrappedStream(baseFn);
    for await (const _item of stream) {
      // drain
    }

    expect(partialToolCall.arguments).toEqual({ path: "/etc/hosts" });
    expect(streamedToolCall.arguments).toEqual({});
  });
});

describe("decodeHtmlEntitiesInObject", () => {
  it("decodes HTML entities in string values", () => {
    const result = decodeHtmlEntitiesInObject(
      "source .env &amp;&amp; psql &quot;$DB&quot; -c &lt;query&gt;",
    );
    expect(result).toBe('source .env && psql "$DB" -c <query>');
  });

  it("recursively decodes nested objects", () => {
    const input = {
      command: "cd ~/dev &amp;&amp; npm run build",
      args: ["--flag=&quot;value&quot;", "&lt;input&gt;"],
      nested: { deep: "a &amp; b" },
    };
    const result = decodeHtmlEntitiesInObject(input) as Record<string, unknown>;
    expect(result.command).toBe("cd ~/dev && npm run build");
    expect((result.args as string[])[0]).toBe('--flag="value"');
    expect((result.args as string[])[1]).toBe("<input>");
    expect((result.nested as Record<string, string>).deep).toBe("a & b");
  });

  it("passes through non-string primitives unchanged", () => {
    expect(decodeHtmlEntitiesInObject(42)).toBe(42);
    expect(decodeHtmlEntitiesInObject(null)).toBe(null);
    expect(decodeHtmlEntitiesInObject(true)).toBe(true);
    expect(decodeHtmlEntitiesInObject(undefined)).toBe(undefined);
  });

  it("returns strings without entities unchanged", () => {
    const input = "plain string with no entities";
    expect(decodeHtmlEntitiesInObject(input)).toBe(input);
  });

  it("decodes numeric character references", () => {
    expect(decodeHtmlEntitiesInObject("&#39;hello&#39;")).toBe("'hello'");
    expect(decodeHtmlEntitiesInObject("&#x27;world&#x27;")).toBe("'world'");
  });
});
describe("prependSystemPromptAddition", () => {
  it("prepends context-engine addition to the system prompt", () => {
    const result = prependSystemPromptAddition({
      systemPrompt: "base system",
      systemPromptAddition: "extra behavior",
    });

    expect(result).toBe("extra behavior\n\nbase system");
  });

  it("returns the original system prompt when no addition is provided", () => {
    const result = prependSystemPromptAddition({
      systemPrompt: "base system",
    });

    expect(result).toBe("base system");
  });
});

describe("buildAfterTurnRuntimeContext", () => {
  it("uses primary model when compaction.model is not set", () => {
    const legacy = buildAfterTurnRuntimeContext({
      attempt: {
        sessionKey: "agent:main:session:abc",
        messageChannel: "slack",
        messageProvider: "slack",
        agentAccountId: "acct-1",
        authProfileId: "openai:p1",
        config: {} as OpenClawConfig,
        skillsSnapshot: undefined,
        senderIsOwner: true,
        provider: "openai-codex",
        modelId: "gpt-5.4",
        thinkLevel: "off",
        reasoningLevel: "on",
        extraSystemPrompt: "extra",
        ownerNumbers: ["+15555550123"],
      },
      workspaceDir: "/tmp/workspace",
      agentDir: "/tmp/agent",
    });

    expect(legacy).toMatchObject({
      provider: "openai-codex",
      model: "gpt-5.4",
    });
  });

  it("resolves compaction.model override in runtime context so all context engines use the correct model", () => {
    const legacy = buildAfterTurnRuntimeContext({
      attempt: {
        sessionKey: "agent:main:session:abc",
        messageChannel: "slack",
        messageProvider: "slack",
        agentAccountId: "acct-1",
        authProfileId: "openai:p1",
        config: {
          agents: {
            defaults: {
              compaction: {
                model: "openrouter/anthropic/claude-sonnet-4-5",
              },
            },
          },
        } as OpenClawConfig,
        skillsSnapshot: undefined,
        senderIsOwner: true,
        provider: "openai-codex",
        modelId: "gpt-5.4",
        thinkLevel: "off",
        reasoningLevel: "on",
        extraSystemPrompt: "extra",
        ownerNumbers: ["+15555550123"],
      },
      workspaceDir: "/tmp/workspace",
      agentDir: "/tmp/agent",
    });

    // buildEmbeddedCompactionRuntimeContext now resolves the override eagerly
    // so that context engines (including third-party ones) receive the correct
    // compaction model in the runtime context.
    expect(legacy).toMatchObject({
      provider: "openrouter",
      model: "anthropic/claude-sonnet-4-5",
      // Auth profile dropped because provider changed from openai-codex to openrouter
      authProfileId: undefined,
    });
  });
  it("includes resolved auth profile fields for context-engine afterTurn compaction", () => {
    const promptCache = buildContextEnginePromptCacheInfo({
      lastCallUsage: {
        input: 10,
        output: 5,
        cacheRead: 40,
        cacheWrite: 2,
        total: 57,
      },
    });
    const legacy = buildAfterTurnRuntimeContext({
      attempt: {
        sessionKey: "agent:main:session:abc",
        messageChannel: "slack",
        messageProvider: "slack",
        agentAccountId: "acct-1",
        authProfileId: "openai:p1",
        config: { plugins: { slots: { contextEngine: "lossless-claw" } } } as OpenClawConfig,
        skillsSnapshot: undefined,
        senderIsOwner: true,
        provider: "openai-codex",
        modelId: "gpt-5.4",
        thinkLevel: "off",
        reasoningLevel: "on",
        extraSystemPrompt: "extra",
        ownerNumbers: ["+15555550123"],
      },
      workspaceDir: "/tmp/workspace",
      agentDir: "/tmp/agent",
      tokenBudget: 1050000,
      currentTokenCount: 52,
      promptCache,
    });

    expect(legacy).toMatchObject({
      authProfileId: "openai:p1",
      provider: "openai-codex",
      model: "gpt-5.4",
      workspaceDir: "/tmp/workspace",
      agentDir: "/tmp/agent",
      tokenBudget: 1050000,
      currentTokenCount: 52,
      promptCache: {
        lastCallUsage: {
          total: 57,
        },
      },
    });
  });

  it("derives afterTurn token count from the current assistant usage snapshot", () => {
    const lastCallUsage = {
      input: 10,
      output: 5,
      cacheRead: 40,
      cacheWrite: 2,
      total: 57,
    };
    const promptCache = buildContextEnginePromptCacheInfo({ lastCallUsage });
    const legacy = buildAfterTurnRuntimeContextFromUsage({
      attempt: {
        sessionKey: "agent:main:session:abc",
        messageChannel: "slack",
        messageProvider: "slack",
        agentAccountId: "acct-1",
        authProfileId: "openai:p1",
        config: { plugins: { slots: { contextEngine: "lossless-claw" } } } as OpenClawConfig,
        skillsSnapshot: undefined,
        senderIsOwner: true,
        provider: "openai-codex",
        modelId: "gpt-5.4",
        thinkLevel: "off",
        reasoningLevel: "on",
        extraSystemPrompt: "extra",
        ownerNumbers: ["+15555550123"],
      },
      workspaceDir: "/tmp/workspace",
      agentDir: "/tmp/agent",
      tokenBudget: 1050000,
      lastCallUsage,
      promptCache,
    });

    expect(legacy).toMatchObject({
      currentTokenCount: 52,
      promptCache: {
        lastCallUsage: {
          total: 57,
        },
      },
    });
  });

  it("preserves sender and channel routing context for scoped compaction discovery", () => {
    const legacy = buildAfterTurnRuntimeContext({
      attempt: {
        sessionKey: "agent:main:session:abc",
        messageChannel: "slack",
        messageProvider: "slack",
        agentAccountId: "acct-1",
        currentChannelId: "C123",
        currentThreadTs: "thread-9",
        currentMessageId: "msg-42",
        authProfileId: "openai:p1",
        config: {} as OpenClawConfig,
        skillsSnapshot: undefined,
        senderIsOwner: true,
        senderId: "user-123",
        provider: "openai-codex",
        modelId: "gpt-5.4",
        thinkLevel: "off",
        reasoningLevel: "on",
        extraSystemPrompt: "extra",
        ownerNumbers: ["+15555550123"],
      },
      workspaceDir: "/tmp/workspace",
      agentDir: "/tmp/agent",
    });

    expect(legacy).toMatchObject({
      senderId: "user-123",
      currentChannelId: "C123",
      currentThreadTs: "thread-9",
      currentMessageId: "msg-42",
    });
  });
});

// ---------------------------------------------------------------------------
// isDeepSeekOnAzureFoundry
// ---------------------------------------------------------------------------
describe("isDeepSeekOnAzureFoundry", () => {
  it("returns true for DeepSeek model on Azure AI Foundry", () => {
    expect(
      isDeepSeekOnAzureFoundry(
        "DeepSeek-R1-0528",
        "https://my-project.services.ai.azure.com/openai/deployments/DeepSeek-R1-0528",
      ),
    ).toBe(true);
  });

  it("returns true for deepseek-v3 on classic Azure OpenAI", () => {
    expect(isDeepSeekOnAzureFoundry("deepseek-v3.2", "https://my-resource.openai.azure.com")).toBe(
      true,
    );
  });

  it("returns false for non-Azure base URL", () => {
    expect(isDeepSeekOnAzureFoundry("deepseek-v3.2", "https://qianfan.baidubce.com/v2")).toBe(
      false,
    );
  });

  it("returns false for non-DeepSeek model on Azure", () => {
    expect(
      isDeepSeekOnAzureFoundry(
        "gpt-4o",
        "https://my-project.services.ai.azure.com/openai/deployments/gpt-4o",
      ),
    ).toBe(false);
  });

  it("returns false when modelId is undefined", () => {
    expect(isDeepSeekOnAzureFoundry(undefined, "https://my-project.services.ai.azure.com")).toBe(
      false,
    );
  });

  it("returns false when baseUrl is undefined", () => {
    expect(isDeepSeekOnAzureFoundry("deepseek-v3.2", undefined)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// extractInlineToolCall
// ---------------------------------------------------------------------------
describe("extractInlineToolCall", () => {
  it("extracts a standalone inline tool call", () => {
    const result = extractInlineToolCall(
      'exec{"command": "curl -s http://example.com"}',
      new Set(["exec"]),
    );
    expect(result).toEqual({
      name: "exec",
      arguments: { command: "curl -s http://example.com" },
      precedingText: "",
    });
  });

  it("extracts tool call with nested JSON", () => {
    const result = extractInlineToolCall(
      'read{"file_path": "/tmp/foo.txt", "options": {"encoding": "utf8"}}',
      new Set(["read"]),
    );
    expect(result).toEqual({
      name: "read",
      arguments: { file_path: "/tmp/foo.txt", options: { encoding: "utf8" } },
      precedingText: "",
    });
  });

  it("handles surrounding whitespace", () => {
    const result = extractInlineToolCall('  exec{"command": "ls"}  ', new Set(["exec"]));
    expect(result).toEqual({
      name: "exec",
      arguments: { command: "ls" },
      precedingText: "",
    });
  });

  it("extracts tool call preceded by conversational text", () => {
    const result = extractInlineToolCall(
      'Let me check the state first.\n    exec{"command": "curl -s http://example.com"}',
      new Set(["exec"]),
    );
    expect(result).toEqual({
      name: "exec",
      arguments: { command: "curl -s http://example.com" },
      precedingText: "Let me check the state first.",
    });
  });

  it("extracts tool call with multi-line preceding text", () => {
    const input =
      'Yes, let\'s try turning them on again with different parameters.\nFirst, check the state.\n    exec{"command": "curl -s http://ha/api/states/light.twinkly | jq ."}';
    const result = extractInlineToolCall(input, new Set(["exec"]));
    expect(result).not.toBeNull();
    expect(result!.name).toBe("exec");
    expect(result!.precedingText).toBe(
      "Yes, let's try turning them on again with different parameters.\nFirst, check the state.",
    );
  });

  it("returns null for unknown tool names", () => {
    expect(
      extractInlineToolCall('unknownTool{"arg": "value"}', new Set(["exec", "read"])),
    ).toBeNull();
  });

  it("extracts without allowedToolNames filter", () => {
    const result = extractInlineToolCall('exec{"command": "ls"}');
    expect(result).toEqual({
      name: "exec",
      arguments: { command: "ls" },
      precedingText: "",
    });
  });

  it("returns null for invalid JSON", () => {
    expect(extractInlineToolCall("exec{not valid json}", new Set(["exec"]))).toBeNull();
  });

  it("returns null for plain text without tool call pattern", () => {
    expect(
      extractInlineToolCall("Let me check the lights for you and report back.", new Set(["exec"])),
    ).toBeNull();
  });

  it("returns null for prose containing curly braces with unknown tool name", () => {
    expect(
      extractInlineToolCall(
        'The config uses format{"key": "value"} for settings.',
        new Set(["exec", "read"]),
      ),
    ).toBeNull();
  });

  it("returns null for JSON arrays", () => {
    expect(extractInlineToolCall("exec[1, 2, 3]", new Set(["exec"]))).toBeNull();
  });

  it("handles control characters around tool name (DeepSeek Azure quirk)", () => {
    // DeepSeek on Azure Foundry inserts control chars like \u0017 (ETB) and \u0015 (NAK)
    const input =
      'The API is responding.\n\n        \u0017exec\u0015{"command": "curl -s http://example.com"}';
    const result = extractInlineToolCall(input, new Set(["exec"]));
    expect(result).not.toBeNull();
    expect(result!.name).toBe("exec");
    expect(result!.arguments).toEqual({ command: "curl -s http://example.com" });
    expect(result!.precedingText).toBe("The API is responding.");
  });

  it("extracts function-call style with parentheses: exec({...})", () => {
    const input =
      'Let me check the state.\n\n          exec({"command": "curl -s http://example.com"})';
    const result = extractInlineToolCall(input, new Set(["exec"]));
    expect(result).not.toBeNull();
    expect(result!.name).toBe("exec");
    expect(result!.arguments).toEqual({ command: "curl -s http://example.com" });
    expect(result!.precedingText).toBe("Let me check the state.");
  });

  it("extracts standalone function-call style: exec({...})", () => {
    const result = extractInlineToolCall('exec({"command": "ls -la"})', new Set(["exec"]));
    expect(result).toEqual({
      name: "exec",
      arguments: { command: "ls -la" },
      precedingText: "",
    });
  });

  // DeepSeek V3.2 standalone JSON patterns
  it("extracts double-brace JSON with action field", () => {
    const input =
      'Let me search memory.\n\n{{"action": "memory_search", "query": "twinkly lights"}}';
    const result = extractInlineToolCall(input, new Set(["memory_search", "exec"]));
    expect(result).toEqual({
      name: "memory_search",
      arguments: { query: "twinkly lights" },
      precedingText: "Let me search memory.",
    });
  });

  it("extracts single-brace JSON with action field", () => {
    const input = 'Checking.\n\n{"action": "exec", "command": "ls -la"}';
    const result = extractInlineToolCall(input, new Set(["exec"]));
    expect(result).toEqual({
      name: "exec",
      arguments: { command: "ls -la" },
      precedingText: "Checking.",
    });
  });

  it("extracts standalone JSON by unique argument key: command → exec", () => {
    const input = 'Let me run that.\n\n{"command": "curl -s http://ha/api/states"}';
    const result = extractInlineToolCall(input, new Set(["exec", "read"]));
    expect(result).toEqual({
      name: "exec",
      arguments: { command: "curl -s http://ha/api/states" },
      precedingText: "Let me run that.",
    });
  });

  it("extracts standalone JSON by unique argument key: path → read", () => {
    const input = 'Let me read the skill file.\n\n{"path": "~/.openclaw/skills/ha/SKILL.md"}';
    const result = extractInlineToolCall(input, new Set(["exec", "read"]));
    expect(result).toEqual({
      name: "read",
      arguments: { path: "~/.openclaw/skills/ha/SKILL.md" },
      precedingText: "Let me read the skill file.",
    });
  });

  it("returns null for standalone JSON with ambiguous keys (query)", () => {
    const input = 'Searching.\n\n{"query": "twinkly lights"}';
    // query is shared by memory_search and web_search — ambiguous
    const result = extractInlineToolCall(input, new Set(["memory_search", "web_search"]));
    expect(result).toBeNull();
  });

  it("extracts last JSON block when multiple are present", () => {
    const input =
      'Trying tool.\n{ "query": "lights" }\n\n{{"action": "memory_search", "query": "lights"}}';
    const result = extractInlineToolCall(input, new Set(["memory_search"]));
    expect(result).not.toBeNull();
    expect(result!.name).toBe("memory_search");
    expect(result!.arguments).toEqual({ query: "lights" });
  });

  it("returns null for standalone JSON when tool not in allowed set", () => {
    const input = '{"action": "memory_search", "query": "test"}';
    const result = extractInlineToolCall(input, new Set(["exec"]));
    expect(result).toBeNull();
  });

  it("extracts standalone JSON with no preceding text", () => {
    const result = extractInlineToolCall('{"action": "exec", "command": "ls"}', new Set(["exec"]));
    expect(result).toEqual({
      name: "exec",
      arguments: { command: "ls" },
      precedingText: "",
    });
  });

  it("extracts JSON tool call when garbled text follows the JSON block", () => {
    // DeepSeek V3.2 sometimes outputs garbled/repeated text after the JSON tool call.
    // The extraction must find the JSON block by matching braces, not by assuming
    // the JSON extends to the end of the text.
    const input =
      "I'll follow the Home Assistant skill. First, let me read it to see how to query lights.\n\n" +
      '{ "path": "~/.openclaw/skills/home-assistant/SKILL.md" }' +
      "            I'll follow the Home Assistant skill. First, let me read it" +
      '.": "~/.openclaw/skills/home-assistant/SKILL.md" }';
    const result = extractInlineToolCall(input, new Set(["exec", "read", "web_search"]));
    expect(result).not.toBeNull();
    expect(result!.name).toBe("read");
    expect(result!.arguments).toEqual({ path: "~/.openclaw/skills/home-assistant/SKILL.md" });
    expect(result!.precedingText).toBe(
      "I'll follow the Home Assistant skill. First, let me read it to see how to query lights.",
    );
  });

  it("extracts double-brace JSON when garbled text follows", () => {
    const input =
      'Searching memory.\n\n{{"action": "memory_search", "query": "lights"}}  some garbled text here}';
    const result = extractInlineToolCall(input, new Set(["memory_search"]));
    expect(result).not.toBeNull();
    expect(result!.name).toBe("memory_search");
    expect(result!.arguments).toEqual({ query: "lights" });
  });
});

// ---------------------------------------------------------------------------
// wrapStreamFnExtractDeepSeekInlineToolCalls
// ---------------------------------------------------------------------------
describe("wrapStreamFnExtractDeepSeekInlineToolCalls", () => {
  function createFakeStream(params: { events: unknown[]; resultMessage: unknown }): {
    result: () => Promise<unknown>;
    [Symbol.asyncIterator]: () => AsyncIterator<unknown>;
  } {
    return {
      async result() {
        return params.resultMessage;
      },
      [Symbol.asyncIterator]() {
        return (async function* () {
          for (const event of params.events) {
            yield event;
          }
        })();
      },
    };
  }

  async function invokeWrappedStream(
    baseFn: (...args: never[]) => unknown,
    allowedToolNames?: Set<string>,
  ) {
    const wrappedFn = wrapStreamFnExtractDeepSeekInlineToolCalls(baseFn as never, allowedToolNames);
    return await wrappedFn({} as never, {} as never, {} as never);
  }

  it("converts inline text tool call to structured tool call in result message", async () => {
    const textBlock = { type: "text", text: 'exec{"command": "ls -la"}' };
    const finalMessage = { role: "assistant", content: [textBlock] };
    const baseFn = vi.fn(() => createFakeStream({ events: [], resultMessage: finalMessage }));

    const stream = await invokeWrappedStream(baseFn, new Set(["exec"]));
    const result = (await stream.result()) as unknown as {
      content: Array<Record<string, unknown>>;
    };

    expect(result.content).toHaveLength(1);
    expect(result.content[0].type).toBe("toolCall");
    expect(result.content[0].name).toBe("exec");
    expect(result.content[0].arguments).toEqual({ command: "ls -la" });
    expect(result.content[0].id).toMatch(/^call_ds_inline_/);
  });

  it("converts inline tool call in streamed message events", async () => {
    const textBlock = { type: "text", text: 'read{"file_path": "/tmp/test.txt"}' };
    const event = {
      type: "content_delta",
      message: { role: "assistant", content: [textBlock] },
    };
    const finalMessage = { role: "assistant", content: [] };
    const baseFn = vi.fn(() => createFakeStream({ events: [event], resultMessage: finalMessage }));

    const stream = await invokeWrappedStream(baseFn, new Set(["read"]));

    const seenEvents: unknown[] = [];
    for await (const item of stream) {
      seenEvents.push(item);
    }

    const eventMsg = (seenEvents[0] as { message: { content: Array<Record<string, unknown>> } })
      .message;
    expect(eventMsg.content).toHaveLength(1);
    expect(eventMsg.content[0].type).toBe("toolCall");
    expect(eventMsg.content[0].name).toBe("read");
  });

  it("preserves normal text blocks that are not tool calls", async () => {
    const textBlock = { type: "text", text: "Let me help you with that." };
    const finalMessage = { role: "assistant", content: [textBlock] };
    const baseFn = vi.fn(() => createFakeStream({ events: [], resultMessage: finalMessage }));

    const stream = await invokeWrappedStream(baseFn, new Set(["exec"]));
    const result = (await stream.result()) as unknown as {
      content: Array<Record<string, unknown>>;
    };

    expect(result.content).toHaveLength(1);
    expect(result.content[0].type).toBe("text");
    expect(result.content[0].text).toBe("Let me help you with that.");
  });

  it("skips extraction when structured tool calls already exist", async () => {
    const textBlock = { type: "text", text: 'exec{"command": "ls"}' };
    const toolCallBlock = {
      type: "toolCall",
      id: "call_existing",
      name: "read",
      arguments: { file_path: "/tmp/foo" },
    };
    const finalMessage = { role: "assistant", content: [textBlock, toolCallBlock] };
    const baseFn = vi.fn(() => createFakeStream({ events: [], resultMessage: finalMessage }));

    const stream = await invokeWrappedStream(baseFn, new Set(["exec", "read"]));
    const result = (await stream.result()) as unknown as {
      content: Array<Record<string, unknown>>;
    };

    // Should keep original content unchanged since structured tool calls exist
    expect(result.content).toHaveLength(2);
    expect(result.content[0].type).toBe("text");
    expect(result.content[1].type).toBe("toolCall");
  });

  it("supports async stream functions that return a promise", async () => {
    const textBlock = { type: "text", text: 'exec{"command": "whoami"}' };
    const finalMessage = { role: "assistant", content: [textBlock] };
    const baseFn = vi.fn(async () => createFakeStream({ events: [], resultMessage: finalMessage }));

    const stream = await invokeWrappedStream(baseFn, new Set(["exec"]));
    const result = (await stream.result()) as unknown as {
      content: Array<Record<string, unknown>>;
    };

    expect(result.content).toHaveLength(1);
    expect(result.content[0].type).toBe("toolCall");
    expect(result.content[0].name).toBe("exec");
  });

  it("splits mixed text + inline tool call into text block + tool call block", async () => {
    const textBlock = {
      type: "text",
      text: 'Let me check the state first.\n    exec{"command": "curl -s http://ha/api/states/light.twinkly | jq ."}',
    };
    const finalMessage = { role: "assistant", content: [textBlock] };
    const baseFn = vi.fn(() => createFakeStream({ events: [], resultMessage: finalMessage }));

    const stream = await invokeWrappedStream(baseFn, new Set(["exec"]));
    const result = (await stream.result()) as unknown as {
      content: Array<Record<string, unknown>>;
    };

    expect(result.content).toHaveLength(2);
    expect(result.content[0].type).toBe("text");
    expect(result.content[0].text).toBe("Let me check the state first.");
    expect(result.content[1].type).toBe("toolCall");
    expect(result.content[1].name).toBe("exec");
    expect(result.content[1].arguments).toEqual({
      command: "curl -s http://ha/api/states/light.twinkly | jq .",
    });
  });

  it("strips inline tool call text from partial events (control chars)", async () => {
    const partialContent = [
      { type: "text", text: 'Let me check.\n\n  \u0017exec\u0015{"command":' },
    ];
    const event = {
      type: "text_delta",
      partial: { role: "assistant", content: partialContent },
    };
    const finalMessage = {
      role: "assistant",
      content: [{ type: "text", text: 'Let me check.\n\n  \u0017exec\u0015{"command": "ls"}' }],
    };
    const baseFn = vi.fn(() => createFakeStream({ events: [event], resultMessage: finalMessage }));

    const stream = await invokeWrappedStream(baseFn, new Set(["exec"]));

    const seenEvents: unknown[] = [];
    for await (const item of stream) {
      seenEvents.push(item);
    }

    // Partial should have tool call text stripped (truncated at control char)
    const partialMsg = (seenEvents[0] as { partial: { content: Array<{ text: string }> } }).partial;
    expect(partialMsg.content[0].text).toBe("Let me check.");

    // Final message should have proper tool call extraction
    const result = (await stream.result()) as unknown as {
      content: Array<Record<string, unknown>>;
    };
    expect(result.content).toHaveLength(2);
    expect(result.content[0].type).toBe("text");
    expect(result.content[1].type).toBe("toolCall");
  });

  it("strips partial inline tool call text without control chars", async () => {
    const partialContent = [{ type: "text", text: 'Checking now.\n  exec{"command": "ls -' }];
    const event = {
      type: "text_delta",
      partial: { role: "assistant", content: partialContent },
    };
    const finalMessage = {
      role: "assistant",
      content: [{ type: "text", text: 'Checking now.\n  exec{"command": "ls -la"}' }],
    };
    const baseFn = vi.fn(() => createFakeStream({ events: [event], resultMessage: finalMessage }));

    const stream = await invokeWrappedStream(baseFn, new Set(["exec"]));

    const seenEvents: unknown[] = [];
    for await (const item of stream) {
      seenEvents.push(item);
    }

    // Partial should have tool call text stripped (truncated at tool name)
    const partialMsg = (seenEvents[0] as { partial: { content: Array<{ text: string }> } }).partial;
    expect(partialMsg.content[0].text).toBe("Checking now.");
  });

  it("does not strip partial text when no tool name matches", async () => {
    const partialContent = [{ type: "text", text: 'Here is some code: config{"key": "value' }];
    const event = {
      type: "text_delta",
      partial: { role: "assistant", content: partialContent },
    };
    const finalMessage = {
      role: "assistant",
      content: [{ type: "text", text: 'Here is some code: config{"key": "value"}' }],
    };
    const baseFn = vi.fn(() => createFakeStream({ events: [event], resultMessage: finalMessage }));

    const stream = await invokeWrappedStream(baseFn, new Set(["exec"]));

    const seenEvents: unknown[] = [];
    for await (const item of stream) {
      seenEvents.push(item);
    }

    // Partial should be preserved since "config" is not in allowedToolNames
    const partialMsg = (seenEvents[0] as { partial: { content: Array<{ text: string }> } }).partial;
    expect(partialMsg.content[0].text).toBe('Here is some code: config{"key": "value');
  });

  it("strips standalone JSON partial with action key", async () => {
    const partialContent = [
      { type: "text", text: 'Let me search.\n\n{{"action": "memory_search", "query": "twi' },
    ];
    const event = {
      type: "text_delta",
      partial: { role: "assistant", content: partialContent },
    };
    const finalMessage = {
      role: "assistant",
      content: [
        {
          type: "text",
          text: 'Let me search.\n\n{{"action": "memory_search", "query": "twinkly lights"}}',
        },
      ],
    };
    const baseFn = vi.fn(() => createFakeStream({ events: [event], resultMessage: finalMessage }));

    const stream = await invokeWrappedStream(baseFn, new Set(["memory_search"]));

    const seenEvents: unknown[] = [];
    for await (const item of stream) {
      seenEvents.push(item);
    }

    const partialMsg = (seenEvents[0] as { partial: { content: Array<{ text: string }> } }).partial;
    expect(partialMsg.content[0].text).toBe("Let me search.");
  });

  it("strips standalone JSON partial with command key", async () => {
    const partialContent = [{ type: "text", text: 'Running command.\n\n{"command": "curl -s' }];
    const event = {
      type: "text_delta",
      partial: { role: "assistant", content: partialContent },
    };
    const finalMessage = {
      role: "assistant",
      content: [{ type: "text", text: 'Running command.\n\n{"command": "curl -s http://ha/api"}' }],
    };
    const baseFn = vi.fn(() => createFakeStream({ events: [event], resultMessage: finalMessage }));

    const stream = await invokeWrappedStream(baseFn, new Set(["exec"]));

    const seenEvents: unknown[] = [];
    for await (const item of stream) {
      seenEvents.push(item);
    }

    const partialMsg = (seenEvents[0] as { partial: { content: Array<{ text: string }> } }).partial;
    expect(partialMsg.content[0].text).toBe("Running command.");
  });

  it("extracts double-brace tool call from final message", async () => {
    const finalMessage = {
      role: "assistant",
      content: [
        {
          type: "text",
          text: 'Let me search.\n\n{{"action": "memory_search", "query": "twinkly lights"}}',
        },
      ],
    };
    const baseFn = vi.fn(() => createFakeStream({ events: [], resultMessage: finalMessage }));

    const stream = await invokeWrappedStream(baseFn, new Set(["memory_search"]));

    // Drain the iterator
    for await (const _ of stream) {
      /* noop */
    }

    const result = (await stream.result()) as unknown as {
      content: Array<Record<string, unknown>>;
    };
    expect(result.content).toHaveLength(2);
    expect(result.content[0].type).toBe("text");
    expect((result.content[0] as { text: string }).text).toBe("Let me search.");
    expect(result.content[1].type).toBe("toolCall");
    expect(result.content[1].name).toBe("memory_search");
    expect(result.content[1].arguments).toEqual({ query: "twinkly lights" });
  });

  it("does not corrupt shared output when stripping partial tool call text", async () => {
    // streamSimple reuses a single output object for both partial and done events.
    // Stripping partial text must not mutate the original, or the final extraction
    // will find stripped text and fail to extract the tool call.
    const sharedTextBlock = { type: "text" as const, text: "Checking." };
    const sharedOutput = { role: "assistant", content: [sharedTextBlock] };

    const baseFn = vi.fn(() => {
      let eventIndex = 0;
      return {
        async result() {
          return sharedOutput;
        },
        [Symbol.asyncIterator]() {
          return {
            async next() {
              if (eventIndex === 0) {
                sharedTextBlock.text = 'Checking.\nexec{"command": "ls -';
                eventIndex++;
                return { done: false, value: { type: "text_delta", partial: sharedOutput } };
              }
              if (eventIndex === 1) {
                sharedTextBlock.text = 'Checking.\nexec{"command": "ls -la"}';
                eventIndex++;
                return { done: false, value: { type: "text_delta", partial: sharedOutput } };
              }
              return { done: true, value: undefined };
            },
            async return() {
              return { done: true as const, value: undefined };
            },
            async throw() {
              return { done: true as const, value: undefined };
            },
          };
        },
      };
    });

    const stream = await invokeWrappedStream(baseFn, new Set(["exec"]));

    const seenPartialTexts: string[] = [];
    for await (const item of stream) {
      const evt = item as { partial?: { content: Array<{ text: string }> } };
      if (evt.partial) {
        seenPartialTexts.push(evt.partial.content[0].text);
      }
    }

    // Partials should have tool call text stripped for UI.
    // Both partials strip from the tool name (exec) since it's in allowedToolNames.
    expect(seenPartialTexts[0]).toBe("Checking.");
    expect(seenPartialTexts[1]).toBe("Checking.");

    // Original shared output must retain full text so extraction works
    const result = (await stream.result()) as unknown as {
      content: Array<Record<string, unknown>>;
    };
    expect(result.content).toHaveLength(2);
    expect(result.content[0].type).toBe("text");
    expect(result.content[1].type).toBe("toolCall");
    expect(result.content[1].name).toBe("exec");
    expect(result.content[1].arguments).toEqual({ command: "ls -la" });
  });

  it("does not corrupt shared output when stripping standalone JSON partial", async () => {
    const sharedTextBlock = { type: "text" as const, text: "Searching." };
    const sharedOutput = { role: "assistant", content: [sharedTextBlock] };

    const baseFn = vi.fn(() => {
      let eventIndex = 0;
      return {
        async result() {
          return sharedOutput;
        },
        [Symbol.asyncIterator]() {
          return {
            async next() {
              if (eventIndex === 0) {
                sharedTextBlock.text = 'Searching.\n\n{"path": "~/.openclaw/skills/ha/SKILL.md';
                eventIndex++;
                return { done: false, value: { type: "text_delta", partial: sharedOutput } };
              }
              if (eventIndex === 1) {
                sharedTextBlock.text = 'Searching.\n\n{"path": "~/.openclaw/skills/ha/SKILL.md"}';
                eventIndex++;
                return { done: false, value: { type: "text_delta", partial: sharedOutput } };
              }
              return { done: true, value: undefined };
            },
            async return() {
              return { done: true as const, value: undefined };
            },
            async throw() {
              return { done: true as const, value: undefined };
            },
          };
        },
      };
    });

    const stream = await invokeWrappedStream(baseFn, new Set(["read"]));
    for await (const _ of stream) {
      /* drain */
    }

    const result = (await stream.result()) as unknown as {
      content: Array<Record<string, unknown>>;
    };
    expect(result.content).toHaveLength(2);
    expect(result.content[0].type).toBe("text");
    expect(result.content[1].type).toBe("toolCall");
    expect(result.content[1].name).toBe("read");
    expect(result.content[1].arguments).toEqual({
      path: "~/.openclaw/skills/ha/SKILL.md",
    });
  });

  it("strips code-fenced tool calls including the fence during streaming", async () => {
    const sharedTextBlock = { type: "text" as const, text: "Reading." };
    const sharedOutput = { role: "assistant", content: [sharedTextBlock] };

    const baseFn = vi.fn(() => {
      let eventIndex = 0;
      return {
        async result() {
          return sharedOutput;
        },
        [Symbol.asyncIterator]() {
          return {
            async next() {
              if (eventIndex === 0) {
                sharedTextBlock.text =
                  'Reading.\n\n```json\n{"action": "read", "file_path": "/tmp/f';
                eventIndex++;
                return { done: false, value: { type: "text_delta", partial: sharedOutput } };
              }
              if (eventIndex === 1) {
                sharedTextBlock.text =
                  'Reading.\n\n```json\n{"action": "read", "file_path": "/tmp/foo.txt"}\n```';
                eventIndex++;
                return { done: false, value: { type: "text_delta", partial: sharedOutput } };
              }
              return { done: true, value: undefined };
            },
            async return() {
              return { done: true as const, value: undefined };
            },
            async throw() {
              return { done: true as const, value: undefined };
            },
          };
        },
      };
    });

    const stream = await invokeWrappedStream(baseFn, new Set(["read"]));

    const seenPartialTexts: string[] = [];
    for await (const item of stream) {
      const evt = item as { partial?: { content: Array<{ text: string }> } };
      if (evt.partial) {
        seenPartialTexts.push(evt.partial.content[0].text);
      }
    }

    // Both partials should strip the code fence + JSON entirely.
    expect(seenPartialTexts[0]).toBe("Reading.");
    expect(seenPartialTexts[1]).toBe("Reading.");

    // Extraction should still produce a toolCall from the unmodified original.
    const result = (await stream.result()) as unknown as {
      content: Array<Record<string, unknown>>;
    };
    expect(result.content).toHaveLength(2);
    expect(result.content[0].type).toBe("text");
    // precedingText should also have the code fence stripped
    expect(result.content[0].text).toBe("Reading.");
    expect(result.content[1].type).toBe("toolCall");
    expect(result.content[1].name).toBe("read");
    expect(result.content[1].arguments).toEqual({ file_path: "/tmp/foo.txt" });
  });
});
