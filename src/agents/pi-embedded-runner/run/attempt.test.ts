import { describe, expect, it, vi } from "vitest";
import type { OpenClawConfig } from "../../../config/config.js";
import { resolveOllamaBaseUrlForRun } from "../../ollama-stream.js";
import { isDeepSeekOnAzureFoundry } from "../model.js";
import {
  buildAfterTurnLegacyCompactionParams,
  composeSystemPromptWithHookContext,
  extractInlineToolCall,
  isOllamaCompatProvider,
  prependSystemPromptAddition,
  resolveAttemptFsWorkspaceOnly,
  resolveOllamaCompatNumCtxEnabled,
  resolvePromptBuildHookResult,
  resolvePromptModeForSession,
  shouldInjectOllamaCompatNumCtx,
  decodeHtmlEntitiesInObject,
  wrapOllamaCompatNumCtx,
  wrapStreamFnExtractDeepSeekInlineToolCalls,
  wrapStreamFnTrimToolCallNames,
} from "./attempt.js";

function createOllamaProviderConfig(injectNumCtxForOpenAICompat: boolean): OpenClawConfig {
  return {
    models: {
      providers: {
        ollama: {
          baseUrl: "http://127.0.0.1:11434/v1",
          api: "openai-completions",
          injectNumCtxForOpenAICompat,
          models: [],
        },
      },
    },
  };
}

describe("resolvePromptBuildHookResult", () => {
  function createLegacyOnlyHookRunner() {
    return {
      hasHooks: vi.fn(
        (hookName: "before_prompt_build" | "before_agent_start") =>
          hookName === "before_agent_start",
      ),
      runBeforePromptBuild: vi.fn(async () => undefined),
      runBeforeAgentStart: vi.fn(async () => ({ prependContext: "from-hook" })),
    };
  }

  it("reuses precomputed legacy before_agent_start result without invoking hook again", async () => {
    const hookRunner = createLegacyOnlyHookRunner();
    const result = await resolvePromptBuildHookResult({
      prompt: "hello",
      messages: [],
      hookCtx: {},
      hookRunner,
      legacyBeforeAgentStartResult: { prependContext: "from-cache", systemPrompt: "legacy-system" },
    });

    expect(hookRunner.runBeforeAgentStart).not.toHaveBeenCalled();
    expect(result).toEqual({
      prependContext: "from-cache",
      systemPrompt: "legacy-system",
      prependSystemContext: undefined,
      appendSystemContext: undefined,
    });
  });

  it("calls legacy hook when precomputed result is absent", async () => {
    const hookRunner = createLegacyOnlyHookRunner();
    const messages = [{ role: "user", content: "ctx" }];
    const result = await resolvePromptBuildHookResult({
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
        prependSystemContext: "prompt prepend",
        appendSystemContext: "prompt append",
      })),
      runBeforeAgentStart: vi.fn(async () => ({
        prependContext: "legacy context",
        prependSystemContext: "legacy prepend",
        appendSystemContext: "legacy append",
      })),
    };

    const result = await resolvePromptBuildHookResult({
      prompt: "hello",
      messages: [],
      hookCtx: {},
      hookRunner,
    });

    expect(result.prependContext).toBe("prompt context\n\nlegacy context");
    expect(result.prependSystemContext).toBe("prompt prepend\n\nlegacy prepend");
    expect(result.appendSystemContext).toBe("prompt append\n\nlegacy append");
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

  it("avoids blank separators when base system prompt is empty", () => {
    expect(
      composeSystemPromptWithHookContext({
        baseSystemPrompt: "   ",
        appendSystemContext: "  append only  ",
      }),
    ).toBe("append only");
  });
});

describe("resolvePromptModeForSession", () => {
  it("uses minimal mode for subagent sessions", () => {
    expect(resolvePromptModeForSession("agent:main:subagent:child")).toBe("minimal");
  });

  it("uses full mode for cron sessions", () => {
    expect(resolvePromptModeForSession("agent:main:cron:job-1")).toBe("full");
    expect(resolvePromptModeForSession("agent:main:cron:job-1:run:run-abc")).toBe("full");
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
describe("wrapStreamFnTrimToolCallNames", () => {
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
    const wrappedFn = wrapStreamFnTrimToolCallNames(baseFn as never, allowedToolNames);
    return await wrappedFn({} as never, {} as never, {} as never);
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
});

describe("isOllamaCompatProvider", () => {
  it("detects native ollama provider id", () => {
    expect(
      isOllamaCompatProvider({
        provider: "ollama",
        api: "openai-completions",
        baseUrl: "https://example.com/v1",
      }),
    ).toBe(true);
  });

  it("detects localhost Ollama OpenAI-compatible endpoint", () => {
    expect(
      isOllamaCompatProvider({
        provider: "custom",
        api: "openai-completions",
        baseUrl: "http://127.0.0.1:11434/v1",
      }),
    ).toBe(true);
  });

  it("does not misclassify non-local OpenAI-compatible providers", () => {
    expect(
      isOllamaCompatProvider({
        provider: "custom",
        api: "openai-completions",
        baseUrl: "https://api.openrouter.ai/v1",
      }),
    ).toBe(false);
  });

  it("detects remote Ollama-compatible endpoint when provider id hints ollama", () => {
    expect(
      isOllamaCompatProvider({
        provider: "my-ollama",
        api: "openai-completions",
        baseUrl: "http://ollama-host:11434/v1",
      }),
    ).toBe(true);
  });

  it("detects IPv6 loopback Ollama OpenAI-compatible endpoint", () => {
    expect(
      isOllamaCompatProvider({
        provider: "custom",
        api: "openai-completions",
        baseUrl: "http://[::1]:11434/v1",
      }),
    ).toBe(true);
  });

  it("does not classify arbitrary remote hosts on 11434 without ollama provider hint", () => {
    expect(
      isOllamaCompatProvider({
        provider: "custom",
        api: "openai-completions",
        baseUrl: "http://example.com:11434/v1",
      }),
    ).toBe(false);
  });
});

describe("resolveOllamaBaseUrlForRun", () => {
  it("prefers provider baseUrl over model baseUrl", () => {
    expect(
      resolveOllamaBaseUrlForRun({
        modelBaseUrl: "http://model-host:11434",
        providerBaseUrl: "http://provider-host:11434",
      }),
    ).toBe("http://provider-host:11434");
  });

  it("falls back to model baseUrl when provider baseUrl is missing", () => {
    expect(
      resolveOllamaBaseUrlForRun({
        modelBaseUrl: "http://model-host:11434",
      }),
    ).toBe("http://model-host:11434");
  });

  it("falls back to native default when neither baseUrl is configured", () => {
    expect(resolveOllamaBaseUrlForRun({})).toBe("http://127.0.0.1:11434");
  });
});

describe("wrapOllamaCompatNumCtx", () => {
  it("injects num_ctx and preserves downstream onPayload hooks", () => {
    let payloadSeen: Record<string, unknown> | undefined;
    const baseFn = vi.fn((_model, _context, options) => {
      const payload: Record<string, unknown> = { options: { temperature: 0.1 } };
      options?.onPayload?.(payload);
      payloadSeen = payload;
      return {} as never;
    });
    const downstream = vi.fn();

    const wrapped = wrapOllamaCompatNumCtx(baseFn as never, 202752);
    void wrapped({} as never, {} as never, { onPayload: downstream } as never);

    expect(baseFn).toHaveBeenCalledTimes(1);
    expect((payloadSeen?.options as Record<string, unknown> | undefined)?.num_ctx).toBe(202752);
    expect(downstream).toHaveBeenCalledTimes(1);
  });
});

describe("resolveOllamaCompatNumCtxEnabled", () => {
  it("defaults to true when config is missing", () => {
    expect(resolveOllamaCompatNumCtxEnabled({ providerId: "ollama" })).toBe(true);
  });

  it("defaults to true when provider config is missing", () => {
    expect(
      resolveOllamaCompatNumCtxEnabled({
        config: { models: { providers: {} } },
        providerId: "ollama",
      }),
    ).toBe(true);
  });

  it("returns false when provider flag is explicitly disabled", () => {
    expect(
      resolveOllamaCompatNumCtxEnabled({
        config: createOllamaProviderConfig(false),
        providerId: "ollama",
      }),
    ).toBe(false);
  });
});

describe("shouldInjectOllamaCompatNumCtx", () => {
  it("requires openai-completions adapter", () => {
    expect(
      shouldInjectOllamaCompatNumCtx({
        model: {
          provider: "ollama",
          api: "openai-responses",
          baseUrl: "http://127.0.0.1:11434/v1",
        },
      }),
    ).toBe(false);
  });

  it("respects provider flag disablement", () => {
    expect(
      shouldInjectOllamaCompatNumCtx({
        model: {
          provider: "ollama",
          api: "openai-completions",
          baseUrl: "http://127.0.0.1:11434/v1",
        },
        config: createOllamaProviderConfig(false),
        providerId: "ollama",
      }),
    ).toBe(false);
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

describe("buildAfterTurnLegacyCompactionParams", () => {
  it("uses primary model when compaction.model is not set", () => {
    const legacy = buildAfterTurnLegacyCompactionParams({
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
        modelId: "gpt-5.3-codex",
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
      model: "gpt-5.3-codex",
    });
  });

  it("passes primary model through even when compaction.model is set (override resolved in compactDirect)", () => {
    const legacy = buildAfterTurnLegacyCompactionParams({
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
        modelId: "gpt-5.3-codex",
        thinkLevel: "off",
        reasoningLevel: "on",
        extraSystemPrompt: "extra",
        ownerNumbers: ["+15555550123"],
      },
      workspaceDir: "/tmp/workspace",
      agentDir: "/tmp/agent",
    });

    // buildAfterTurnLegacyCompactionParams no longer resolves the override;
    // compactEmbeddedPiSessionDirect does it centrally for both auto + manual paths.
    expect(legacy).toMatchObject({
      provider: "openai-codex",
      model: "gpt-5.3-codex",
    });
  });

  it("includes resolved auth profile fields for context-engine afterTurn compaction", () => {
    const legacy = buildAfterTurnLegacyCompactionParams({
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
        modelId: "gpt-5.3-codex",
        thinkLevel: "off",
        reasoningLevel: "on",
        extraSystemPrompt: "extra",
        ownerNumbers: ["+15555550123"],
      },
      workspaceDir: "/tmp/workspace",
      agentDir: "/tmp/agent",
    });

    expect(legacy).toMatchObject({
      authProfileId: "openai:p1",
      provider: "openai-codex",
      model: "gpt-5.3-codex",
      workspaceDir: "/tmp/workspace",
      agentDir: "/tmp/agent",
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
