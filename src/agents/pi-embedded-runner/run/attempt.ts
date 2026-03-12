import fs from "node:fs/promises";
import os from "node:os";
import type { AgentMessage, StreamFn } from "@mariozechner/pi-agent-core";
import { streamSimple } from "@mariozechner/pi-ai";
import {
  createAgentSession,
  DefaultResourceLoader,
  SessionManager,
} from "@mariozechner/pi-coding-agent";
import { resolveHeartbeatPrompt } from "../../../auto-reply/heartbeat.js";
import { resolveChannelCapabilities } from "../../../config/channel-capabilities.js";
import type { OpenClawConfig } from "../../../config/config.js";
import { getMachineDisplayName } from "../../../infra/machine-name.js";
import {
  ensureGlobalUndiciEnvProxyDispatcher,
  ensureGlobalUndiciStreamTimeouts,
} from "../../../infra/net/undici-global-dispatcher.js";
import { MAX_IMAGE_BYTES } from "../../../media/constants.js";
import { getGlobalHookRunner } from "../../../plugins/hook-runner-global.js";
import type {
  PluginHookAgentContext,
  PluginHookBeforeAgentStartResult,
  PluginHookBeforePromptBuildResult,
} from "../../../plugins/types.js";
import { isCronSessionKey, isSubagentSessionKey } from "../../../routing/session-key.js";
import { joinPresentTextSegments } from "../../../shared/text/join-segments.js";
import { resolveSignalReactionLevel } from "../../../signal/reaction-level.js";
import { resolveTelegramInlineButtonsScope } from "../../../telegram/inline-buttons.js";
import { resolveTelegramReactionLevel } from "../../../telegram/reaction-level.js";
import { buildTtsSystemPromptHint } from "../../../tts/tts.js";
import { resolveUserPath } from "../../../utils.js";
import { normalizeMessageChannel } from "../../../utils/message-channel.js";
import { isReasoningTagProvider } from "../../../utils/provider-utils.js";
import { resolveOpenClawAgentDir } from "../../agent-paths.js";
import { resolveSessionAgentIds } from "../../agent-scope.js";
import { createAnthropicPayloadLogger } from "../../anthropic-payload-log.js";
import {
  analyzeBootstrapBudget,
  buildBootstrapPromptWarning,
  buildBootstrapTruncationReportMeta,
  buildBootstrapInjectionStats,
} from "../../bootstrap-budget.js";
import { makeBootstrapWarn, resolveBootstrapContextForRun } from "../../bootstrap-files.js";
import { createCacheTrace } from "../../cache-trace.js";
import {
  listChannelSupportedActions,
  resolveChannelMessageToolHints,
} from "../../channel-tools.js";
import { ensureCustomApiRegistered } from "../../custom-api-registry.js";
import { DEFAULT_CONTEXT_TOKENS } from "../../defaults.js";
import { resolveOpenClawDocsPath } from "../../docs-path.js";
import { isTimeoutError } from "../../failover-error.js";
import { resolveImageSanitizationLimits } from "../../image-sanitization.js";
import { resolveModelAuthMode } from "../../model-auth.js";
import { normalizeProviderId, resolveDefaultModelForAgent } from "../../model-selection.js";
import { supportsModelTools } from "../../model-tool-support.js";
import { createConfiguredOllamaStreamFn } from "../../ollama-stream.js";
import { createOpenAIWebSocketStreamFn, releaseWsSession } from "../../openai-ws-stream.js";
import { resolveOwnerDisplaySetting } from "../../owner-display.js";
import {
  downgradeOpenAIFunctionCallReasoningPairs,
  isCloudCodeAssistFormatError,
  resolveBootstrapMaxChars,
  resolveBootstrapPromptTruncationWarningMode,
  resolveBootstrapTotalMaxChars,
  validateAnthropicTurns,
  validateGeminiTurns,
} from "../../pi-embedded-helpers.js";
import { subscribeEmbeddedPiSession } from "../../pi-embedded-subscribe.js";
import { createPreparedEmbeddedPiSettingsManager } from "../../pi-project-settings.js";
import { applyPiAutoCompactionGuard } from "../../pi-settings.js";
import { toClientToolDefinitions } from "../../pi-tool-definition-adapter.js";
import { createOpenClawCodingTools, resolveToolLoopDetectionConfig } from "../../pi-tools.js";
import { resolveSandboxContext } from "../../sandbox.js";
import { resolveSandboxRuntimeStatus } from "../../sandbox/runtime-status.js";
import { isXaiProvider } from "../../schema/clean-for-xai.js";
import { repairSessionFileIfNeeded } from "../../session-file-repair.js";
import { guardSessionManager } from "../../session-tool-result-guard-wrapper.js";
import { sanitizeToolUseResultPairing } from "../../session-transcript-repair.js";
import {
  acquireSessionWriteLock,
  resolveSessionLockMaxHoldFromTimeout,
} from "../../session-write-lock.js";
import { detectRuntimeShell } from "../../shell-utils.js";
import {
  applySkillEnvOverrides,
  applySkillEnvOverridesFromSnapshot,
  resolveSkillsPromptForRun,
} from "../../skills.js";
import { buildSystemPromptParams } from "../../system-prompt-params.js";
import { buildSystemPromptReport } from "../../system-prompt-report.js";
import { sanitizeToolCallIdsForCloudCodeAssist } from "../../tool-call-id.js";
import { resolveEffectiveToolFsWorkspaceOnly } from "../../tool-fs-policy.js";
import { normalizeToolName } from "../../tool-policy.js";
import { resolveTranscriptPolicy } from "../../transcript-policy.js";
import { DEFAULT_BOOTSTRAP_FILENAME } from "../../workspace.js";
import { isRunnerAbortError } from "../abort.js";
import { appendCacheTtlTimestamp, isCacheTtlEligibleProvider } from "../cache-ttl.js";
import type { CompactEmbeddedPiSessionParams } from "../compact.js";
import { buildEmbeddedExtensionFactories } from "../extensions.js";
import { applyExtraParamsToAgent } from "../extra-params.js";
import {
  logToolSchemasForGoogle,
  sanitizeSessionHistory,
  sanitizeToolsForGoogle,
} from "../google.js";
import { getDmHistoryLimitFromSessionKey, limitHistoryTurns } from "../history.js";
import { log } from "../logger.js";
import { buildModelAliasLines, isDeepSeekOnAzureFoundry } from "../model.js";
import {
  clearActiveEmbeddedRun,
  type EmbeddedPiQueueHandle,
  setActiveEmbeddedRun,
} from "../runs.js";
import { buildEmbeddedSandboxInfo } from "../sandbox-info.js";
import { prewarmSessionFile, trackSessionManagerAccess } from "../session-manager-cache.js";
import { prepareSessionManagerForRun } from "../session-manager-init.js";
import { resolveEmbeddedRunSkillEntries } from "../skills-runtime.js";
import {
  applySystemPromptOverrideToSession,
  buildEmbeddedSystemPrompt,
  createSystemPromptOverride,
} from "../system-prompt.js";
import { dropThinkingBlocks } from "../thinking.js";
import { collectAllowedToolNames } from "../tool-name-allowlist.js";
import { installToolResultContextGuard } from "../tool-result-context-guard.js";
import { splitSdkTools } from "../tool-split.js";
import { describeUnknownError, mapThinkingLevel } from "../utils.js";
import { flushPendingToolResultsAfterIdle } from "../wait-for-idle-before-flush.js";
import { waitForCompactionRetryWithAggregateTimeout } from "./compaction-retry-aggregate-timeout.js";
import {
  selectCompactionTimeoutSnapshot,
  shouldFlagCompactionTimeout,
} from "./compaction-timeout.js";
import { pruneProcessedHistoryImages } from "./history-image-prune.js";
import { detectAndLoadPromptImages } from "./images.js";
import type { EmbeddedRunAttemptParams, EmbeddedRunAttemptResult } from "./types.js";

type PromptBuildHookRunner = {
  hasHooks: (hookName: "before_prompt_build" | "before_agent_start") => boolean;
  runBeforePromptBuild: (
    event: { prompt: string; messages: unknown[] },
    ctx: PluginHookAgentContext,
  ) => Promise<PluginHookBeforePromptBuildResult | undefined>;
  runBeforeAgentStart: (
    event: { prompt: string; messages: unknown[] },
    ctx: PluginHookAgentContext,
  ) => Promise<PluginHookBeforeAgentStartResult | undefined>;
};

export function isOllamaCompatProvider(model: {
  provider?: string;
  baseUrl?: string;
  api?: string;
}): boolean {
  const providerId = normalizeProviderId(model.provider ?? "");
  if (providerId === "ollama") {
    return true;
  }
  if (!model.baseUrl) {
    return false;
  }
  try {
    const parsed = new URL(model.baseUrl);
    const hostname = parsed.hostname.toLowerCase();
    const isLocalhost =
      hostname === "localhost" ||
      hostname === "127.0.0.1" ||
      hostname === "::1" ||
      hostname === "[::1]";
    if (isLocalhost && parsed.port === "11434") {
      return true;
    }

    // Allow remote/LAN Ollama OpenAI-compatible endpoints when the provider id
    // itself indicates Ollama usage (e.g. "my-ollama").
    const providerHintsOllama = providerId.includes("ollama");
    const isOllamaPort = parsed.port === "11434";
    const isOllamaCompatPath = parsed.pathname === "/" || /^\/v1\/?$/i.test(parsed.pathname);
    return providerHintsOllama && isOllamaPort && isOllamaCompatPath;
  } catch {
    return false;
  }
}

export function resolveOllamaCompatNumCtxEnabled(params: {
  config?: OpenClawConfig;
  providerId?: string;
}): boolean {
  const providerId = params.providerId?.trim();
  if (!providerId) {
    return true;
  }
  const providers = params.config?.models?.providers;
  if (!providers) {
    return true;
  }
  const direct = providers[providerId];
  if (direct) {
    return direct.injectNumCtxForOpenAICompat ?? true;
  }
  const normalized = normalizeProviderId(providerId);
  for (const [candidateId, candidate] of Object.entries(providers)) {
    if (normalizeProviderId(candidateId) === normalized) {
      return candidate.injectNumCtxForOpenAICompat ?? true;
    }
  }
  return true;
}

export function shouldInjectOllamaCompatNumCtx(params: {
  model: { api?: string; provider?: string; baseUrl?: string };
  config?: OpenClawConfig;
  providerId?: string;
}): boolean {
  // Restrict to the OpenAI-compatible adapter path only.
  if (params.model.api !== "openai-completions") {
    return false;
  }
  if (!isOllamaCompatProvider(params.model)) {
    return false;
  }
  return resolveOllamaCompatNumCtxEnabled({
    config: params.config,
    providerId: params.providerId,
  });
}

export function wrapOllamaCompatNumCtx(baseFn: StreamFn | undefined, numCtx: number): StreamFn {
  const streamFn = baseFn ?? streamSimple;
  return (model, context, options) =>
    streamFn(model, context, {
      ...options,
      onPayload: (payload: unknown) => {
        if (!payload || typeof payload !== "object") {
          return options?.onPayload?.(payload, model);
        }
        const payloadRecord = payload as Record<string, unknown>;
        if (!payloadRecord.options || typeof payloadRecord.options !== "object") {
          payloadRecord.options = {};
        }
        (payloadRecord.options as Record<string, unknown>).num_ctx = numCtx;
        return options?.onPayload?.(payload, model);
      },
    });
}

function normalizeToolCallNameForDispatch(rawName: string, allowedToolNames?: Set<string>): string {
  const trimmed = rawName.trim();
  if (!trimmed) {
    // Keep whitespace-only placeholders unchanged so they do not collapse to
    // empty names (which can later surface as toolName="" loops).
    return rawName;
  }
  if (!allowedToolNames || allowedToolNames.size === 0) {
    return trimmed;
  }

  const candidateNames = new Set<string>([trimmed, normalizeToolName(trimmed)]);
  const normalizedDelimiter = trimmed.replace(/\//g, ".");
  const segments = normalizedDelimiter
    .split(".")
    .map((segment) => segment.trim())
    .filter(Boolean);
  if (segments.length > 1) {
    for (let index = 1; index < segments.length; index += 1) {
      const suffix = segments.slice(index).join(".");
      candidateNames.add(suffix);
      candidateNames.add(normalizeToolName(suffix));
    }
  }

  for (const candidate of candidateNames) {
    if (allowedToolNames.has(candidate)) {
      return candidate;
    }
  }

  for (const candidate of candidateNames) {
    const folded = candidate.toLowerCase();
    let caseInsensitiveMatch: string | null = null;
    for (const name of allowedToolNames) {
      if (name.toLowerCase() !== folded) {
        continue;
      }
      if (caseInsensitiveMatch && caseInsensitiveMatch !== name) {
        return candidate;
      }
      caseInsensitiveMatch = name;
    }
    if (caseInsensitiveMatch) {
      return caseInsensitiveMatch;
    }
  }

  return trimmed;
}

function isToolCallBlockType(type: unknown): boolean {
  return type === "toolCall" || type === "toolUse" || type === "functionCall";
}

function normalizeToolCallIdsInMessage(message: unknown): void {
  if (!message || typeof message !== "object") {
    return;
  }
  const content = (message as { content?: unknown }).content;
  if (!Array.isArray(content)) {
    return;
  }

  const usedIds = new Set<string>();
  for (const block of content) {
    if (!block || typeof block !== "object") {
      continue;
    }
    const typedBlock = block as { type?: unknown; id?: unknown };
    if (!isToolCallBlockType(typedBlock.type) || typeof typedBlock.id !== "string") {
      continue;
    }
    const trimmedId = typedBlock.id.trim();
    if (!trimmedId) {
      continue;
    }
    usedIds.add(trimmedId);
  }

  let fallbackIndex = 1;
  for (const block of content) {
    if (!block || typeof block !== "object") {
      continue;
    }
    const typedBlock = block as { type?: unknown; id?: unknown };
    if (!isToolCallBlockType(typedBlock.type)) {
      continue;
    }
    if (typeof typedBlock.id === "string") {
      const trimmedId = typedBlock.id.trim();
      if (trimmedId) {
        if (typedBlock.id !== trimmedId) {
          typedBlock.id = trimmedId;
        }
        usedIds.add(trimmedId);
        continue;
      }
    }

    let fallbackId = "";
    while (!fallbackId || usedIds.has(fallbackId)) {
      fallbackId = `call_auto_${fallbackIndex++}`;
    }
    typedBlock.id = fallbackId;
    usedIds.add(fallbackId);
  }
}

function trimWhitespaceFromToolCallNamesInMessage(
  message: unknown,
  allowedToolNames?: Set<string>,
): void {
  if (!message || typeof message !== "object") {
    return;
  }
  const content = (message as { content?: unknown }).content;
  if (!Array.isArray(content)) {
    return;
  }
  for (const block of content) {
    if (!block || typeof block !== "object") {
      continue;
    }
    const typedBlock = block as { type?: unknown; name?: unknown };
    if (!isToolCallBlockType(typedBlock.type) || typeof typedBlock.name !== "string") {
      continue;
    }
    const normalized = normalizeToolCallNameForDispatch(typedBlock.name, allowedToolNames);
    if (normalized !== typedBlock.name) {
      typedBlock.name = normalized;
    }
  }
  normalizeToolCallIdsInMessage(message);
}

function wrapStreamTrimToolCallNames(
  stream: ReturnType<typeof streamSimple>,
  allowedToolNames?: Set<string>,
): ReturnType<typeof streamSimple> {
  const originalResult = stream.result.bind(stream);
  stream.result = async () => {
    const message = await originalResult();
    trimWhitespaceFromToolCallNamesInMessage(message, allowedToolNames);
    return message;
  };

  const originalAsyncIterator = stream[Symbol.asyncIterator].bind(stream);
  (stream as { [Symbol.asyncIterator]: typeof originalAsyncIterator })[Symbol.asyncIterator] =
    function () {
      const iterator = originalAsyncIterator();
      return {
        async next() {
          const result = await iterator.next();
          if (!result.done && result.value && typeof result.value === "object") {
            const event = result.value as {
              partial?: unknown;
              message?: unknown;
            };
            trimWhitespaceFromToolCallNamesInMessage(event.partial, allowedToolNames);
            trimWhitespaceFromToolCallNamesInMessage(event.message, allowedToolNames);
          }
          return result;
        },
        async return(value?: unknown) {
          return iterator.return?.(value) ?? { done: true as const, value: undefined };
        },
        async throw(error?: unknown) {
          return iterator.throw?.(error) ?? { done: true as const, value: undefined };
        },
      };
    };

  return stream;
}

export function wrapStreamFnTrimToolCallNames(
  baseFn: StreamFn,
  allowedToolNames?: Set<string>,
): StreamFn {
  return (model, context, options) => {
    const maybeStream = baseFn(model, context, options);
    if (maybeStream && typeof maybeStream === "object" && "then" in maybeStream) {
      return Promise.resolve(maybeStream).then((stream) =>
        wrapStreamTrimToolCallNames(stream, allowedToolNames),
      );
    }
    return wrapStreamTrimToolCallNames(maybeStream, allowedToolNames);
  };
}

// ---------------------------------------------------------------------------
// xAI / Grok: decode HTML entities in tool call arguments
// ---------------------------------------------------------------------------

const HTML_ENTITY_RE = /&(?:amp|lt|gt|quot|apos|#39|#x[0-9a-f]+|#\d+);/i;

function decodeHtmlEntities(value: string): string {
  return value
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&apos;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&#x([0-9a-f]+);/gi, (_, hex) => String.fromCodePoint(Number.parseInt(hex, 16)))
    .replace(/&#(\d+);/gi, (_, dec) => String.fromCodePoint(Number.parseInt(dec, 10)));
}

export function decodeHtmlEntitiesInObject(obj: unknown): unknown {
  if (typeof obj === "string") {
    return HTML_ENTITY_RE.test(obj) ? decodeHtmlEntities(obj) : obj;
  }
  if (Array.isArray(obj)) {
    return obj.map(decodeHtmlEntitiesInObject);
  }
  if (obj && typeof obj === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(obj as Record<string, unknown>)) {
      result[key] = decodeHtmlEntitiesInObject(val);
    }
    return result;
  }
  return obj;
}

function decodeXaiToolCallArgumentsInMessage(message: unknown): void {
  if (!message || typeof message !== "object") {
    return;
  }
  const content = (message as { content?: unknown }).content;
  if (!Array.isArray(content)) {
    return;
  }
  for (const block of content) {
    if (!block || typeof block !== "object") {
      continue;
    }
    const typedBlock = block as { type?: unknown; arguments?: unknown };
    if (typedBlock.type !== "toolCall" || !typedBlock.arguments) {
      continue;
    }
    if (typeof typedBlock.arguments === "object") {
      typedBlock.arguments = decodeHtmlEntitiesInObject(typedBlock.arguments);
    }
  }
}

function wrapStreamDecodeXaiToolCallArguments(
  stream: ReturnType<typeof streamSimple>,
): ReturnType<typeof streamSimple> {
  const originalResult = stream.result.bind(stream);
  stream.result = async () => {
    const message = await originalResult();
    decodeXaiToolCallArgumentsInMessage(message);
    return message;
  };

  const originalAsyncIterator = stream[Symbol.asyncIterator].bind(stream);
  (stream as { [Symbol.asyncIterator]: typeof originalAsyncIterator })[Symbol.asyncIterator] =
    function () {
      const iterator = originalAsyncIterator();
      return {
        async next() {
          const result = await iterator.next();
          if (!result.done && result.value && typeof result.value === "object") {
            const event = result.value as { partial?: unknown; message?: unknown };
            decodeXaiToolCallArgumentsInMessage(event.partial);
            decodeXaiToolCallArgumentsInMessage(event.message);
          }
          return result;
        },
        async return(value?: unknown) {
          return iterator.return?.(value) ?? { done: true as const, value: undefined };
        },
        async throw(error?: unknown) {
          return iterator.throw?.(error) ?? { done: true as const, value: undefined };
        },
      };
    };
  return stream;
}

function wrapStreamFnDecodeXaiToolCallArguments(baseFn: StreamFn): StreamFn {
  return (model, context, options) => {
    const maybeStream = baseFn(model, context, options);
    if (maybeStream && typeof maybeStream === "object" && "then" in maybeStream) {
      return Promise.resolve(maybeStream).then((stream) =>
        wrapStreamDecodeXaiToolCallArguments(stream),
      );
    }
    return wrapStreamDecodeXaiToolCallArguments(maybeStream);
  };
}

// ---------------------------------------------------------------------------
// DeepSeek on Azure Foundry: extract inline tool calls from text content
// ---------------------------------------------------------------------------
// DeepSeek models hosted on Azure AI Foundry sometimes emit tool calls as
// plain text in the response content (e.g. `exec{"command":"..."}`) instead
// of structured `tool_calls`. This wrapper detects those patterns and
// converts them into proper structured tool call blocks.

/**
 * Regex matching an inline tool call emitted as text by DeepSeek.
 * The tool call can appear standalone or at the end of a larger text block
 * (possibly preceded by conversational text on earlier lines).
 * Captures: (1) optional preceding text, (2) tool name, (3) JSON object.
 * Examples:
 *   exec{"command": "ls -la"}
 *   Let me check.\n    exec{"command": "curl ..."}
 *   read{"file_path": "/tmp/foo.txt"}
 *   exec({"command": "ls"})
 */
const INLINE_TOOL_CALL_RE = /^([\s\S]*?)(\w+)\(?(\{[\s\S]*\})\)?\s*$/;

/** Fields that DeepSeek V3.2 may place inside standalone JSON to name the target tool. */
const TOOL_NAME_JSON_FIELDS = ["action", "name", "tool", "function"] as const;

/**
 * Argument keys that uniquely identify a specific tool. Only unambiguous keys
 * are listed (e.g. `query` is shared by memory_search and web_search).
 */
const UNIQUE_ARG_KEY_TO_TOOL: ReadonlyArray<{ key: string; tool: string }> = [
  { key: "command", tool: "exec" },
  { key: "file_path", tool: "read" },
  { key: "path", tool: "read" },
];

/** Strip a trailing markdown code fence from preceding text so the UI
 *  doesn't render a dangling ` ```json ` block after the tool call is removed. */
function stripTrailingCodeFence(text: string): string {
  return text.replace(/```(?:\w+)?\s*$/, "").trimEnd();
}

/**
 * Try to extract an inline tool call from a text content block.
 * Returns null if the text doesn't match the pattern or the JSON is invalid.
 * When extracted, also returns `precedingText` (the conversational text
 * before the tool call) so the caller can preserve it.
 */
export function extractInlineToolCall(
  text: string,
  allowedToolNames?: Set<string>,
): { name: string; arguments: Record<string, unknown>; precedingText: string } | null {
  // DeepSeek on Azure Foundry inserts non-printable control characters
  // (e.g. \u0017 ETB, \u0015 NAK) around tool names. Strip them before matching.
  // Preserve \t (\x09), \n (\x0a), \r (\x0d) as they're meaningful whitespace.
  // eslint-disable-next-line no-control-regex
  const cleaned = text.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g, "");
  const trimmed = cleaned.trim();

  // Pattern 1: toolName{json} or toolName({json}) — DeepSeek V3/V3.1 style
  const match = INLINE_TOOL_CALL_RE.exec(trimmed);
  if (match) {
    const [, preceding, toolName, jsonStr] = match;
    if (toolName && jsonStr && (!allowedToolNames || allowedToolNames.has(toolName))) {
      try {
        const args = JSON.parse(jsonStr) as Record<string, unknown>;
        if (args && typeof args === "object" && !Array.isArray(args)) {
          return {
            name: toolName,
            arguments: args,
            precedingText: stripTrailingCodeFence((preceding ?? "").trim()),
          };
        }
      } catch {
        // Fall through to standalone JSON extraction
      }
    }
  }

  // Pattern 2: Standalone JSON at end of text — DeepSeek V3.2 style.
  // Handles: {{"action": "tool", ...}}, {"action": "tool", ...},
  // {"command": "..."}, {"path": "..."} where keys identify the tool.
  return extractStandaloneJsonToolCall(trimmed, allowedToolNames);
}

/**
 * Find the position of the closing brace that matches the opening brace at
 * `openIdx`, correctly handling nested braces and JSON string literals.
 * Returns -1 if no matching brace is found.
 */
function findMatchingBrace(text: string, openIdx: number): number {
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = openIdx; i < text.length; i++) {
    const ch = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (ch === "\\" && inString) {
      escape = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) {
      continue;
    }
    if (ch === "{") {
      depth++;
    }
    if (ch === "}") {
      depth--;
      if (depth === 0) {
        return i;
      }
    }
  }
  return -1;
}

/**
 * Extract a tool call from standalone JSON anywhere in the text.
 * DeepSeek V3.2 sometimes omits the tool name prefix and instead:
 *   1. Wraps JSON in double braces with an `action` field:
 *      {{"action": "memory_search", "query": "..."}}
 *   2. Outputs bare JSON whose argument keys identify the tool:
 *      {"command": "ls -la"}  →  exec
 *      {"path": "/tmp/foo"}   →  read
 *
 * The model may also output garbled/repeated text AFTER the JSON block,
 * so we find matching braces rather than assuming JSON extends to end-of-text.
 */
function extractStandaloneJsonToolCall(
  text: string,
  allowedToolNames?: Set<string>,
): { name: string; arguments: Record<string, unknown>; precedingText: string } | null {
  let pos = 0;
  for (let attempt = 0; attempt < 20; attempt++) {
    const idx = text.indexOf("{", pos);
    if (idx < 0) {
      break;
    }

    // Double braces: {{...}} — strip outer layer and parse inner JSON
    if (idx + 1 < text.length && text[idx + 1] === "{") {
      const innerClose = findMatchingBrace(text, idx + 1);
      if (innerClose >= 0 && innerClose + 1 < text.length && text[innerClose + 1] === "}") {
        const inner = text.slice(idx + 1, innerClose + 1);
        const result = tryIdentifyToolFromJsonStr(
          inner,
          stripTrailingCodeFence(text.slice(0, idx).trimEnd()),
          allowedToolNames,
        );
        if (result) {
          return result;
        }
        pos = innerClose + 2;
        continue;
      }
    }

    // Single braces: find matching } and try JSON.parse on that substring
    const closeIdx = findMatchingBrace(text, idx);
    if (closeIdx >= 0) {
      const candidate = text.slice(idx, closeIdx + 1);
      const result = tryIdentifyToolFromJsonStr(
        candidate,
        stripTrailingCodeFence(text.slice(0, idx).trimEnd()),
        allowedToolNames,
      );
      if (result) {
        return result;
      }
      pos = closeIdx + 1;
    } else {
      pos = idx + 1;
    }
  }

  return null;
}

/** Try to parse a JSON string and identify which tool it targets. */
function tryIdentifyToolFromJsonStr(
  jsonStr: string,
  precedingText: string,
  allowedToolNames?: Set<string>,
): { name: string; arguments: Record<string, unknown>; precedingText: string } | null {
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(jsonStr) as Record<string, unknown>;
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return null;
  }

  // Strategy 1: look for a tool name field inside the JSON (e.g. "action")
  for (const field of TOOL_NAME_JSON_FIELDS) {
    const value = parsed[field];
    if (typeof value === "string" && value.trim()) {
      const toolName = value.trim();
      if (!allowedToolNames || allowedToolNames.has(toolName)) {
        const rest: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(parsed)) {
          if (k !== field) {
            rest[k] = v;
          }
        }
        return { name: toolName, arguments: rest, precedingText };
      }
    }
  }

  // Strategy 2: match unique argument keys to known tools
  if (allowedToolNames) {
    for (const { key, tool } of UNIQUE_ARG_KEY_TO_TOOL) {
      if (key in parsed && allowedToolNames.has(tool)) {
        return { name: tool, arguments: parsed, precedingText };
      }
    }
  }

  return null;
}

let inlineToolCallCounter = 0;

function extractInlineToolCallsFromMessage(message: unknown, allowedToolNames?: Set<string>): void {
  if (!message || typeof message !== "object") {
    return;
  }
  const msg = message as { content?: unknown };
  const content = msg.content;
  if (!Array.isArray(content)) {
    return;
  }

  // Skip if the message already has structured tool calls.
  const hasStructuredToolCalls = content.some(
    (block: unknown) =>
      block && typeof block === "object" && isToolCallBlockType((block as { type?: unknown }).type),
  );
  if (hasStructuredToolCalls) {
    return;
  }

  const newContent: unknown[] = [];
  let modified = false;

  for (let i = 0; i < content.length; i++) {
    const block = content[i];
    if (!block || typeof block !== "object") {
      newContent.push(block);
      continue;
    }
    const typedBlock = block as { type?: unknown; text?: unknown };
    if (typedBlock.type !== "text" || typeof typedBlock.text !== "string") {
      newContent.push(block);
      continue;
    }

    const extracted = extractInlineToolCall(typedBlock.text, allowedToolNames);
    log.info(
      `DeepSeek inline extraction: text=${JSON.stringify(typedBlock.text.slice(0, 200))} extracted=${extracted ? `${extracted.name}(${JSON.stringify(extracted.arguments).slice(0, 100)})` : "null"}`,
    );
    if (!extracted) {
      newContent.push(block);
      continue;
    }

    // Preserve any conversational text that preceded the inline tool call.
    if (extracted.precedingText) {
      newContent.push({ type: "text", text: extracted.precedingText });
    }

    // Add a structured tool call block in place of the inline text.
    newContent.push({
      type: "toolCall",
      id: `call_ds_inline_${++inlineToolCallCounter}`,
      name: extracted.name,
      arguments: extracted.arguments,
    });
    modified = true;
  }

  if (modified) {
    msg.content = newContent;
  }
}

/**
 * Strip inline tool call text from a partial (streaming) message so the UI
 * never renders raw `exec{...}` or `{"action":"read",...}` text.
 *
 * DeepSeek on Azure wraps inline tool calls in several ways:
 *   1. Control chars (\u0017, \u0015) before the tool name
 *   2. `exec{"command":"..."}` — tool name prefix + JSON
 *   3. ````json\nexec{"command":"..."}` `` ``` `` — code-fenced tool call
 *   4. `{"action":"read","file_path":"..."}` — standalone JSON with tool key
 *   5. ````json\n{"action":"read",...}` `` ``` `` — code-fenced standalone JSON
 *
 * We find the earliest tool call signal, look backward for a markdown code
 * fence, and truncate at whichever comes first.
 */
function stripPartialInlineToolCallText(partial: unknown, allowedToolNames?: Set<string>): void {
  if (!partial || typeof partial !== "object") {
    return;
  }
  const msg = partial as { content?: unknown[] };
  if (!Array.isArray(msg.content)) {
    return;
  }
  for (const block of msg.content) {
    if (!block || typeof block !== "object") {
      continue;
    }
    const typed = block as { type?: string; text?: string };
    if (typed.type !== "text" || typeof typed.text !== "string") {
      continue;
    }
    // Control chars are the strongest signal — truncate at the first one.
    // eslint-disable-next-line no-control-regex
    const controlIdx = typed.text.search(/[\x00-\x08\x0b\x0c\x0e-\x1f]/);
    if (controlIdx >= 0) {
      typed.text = typed.text.slice(0, controlIdx).trimEnd();
      continue;
    }

    // Find the earliest tool call start position.
    let stripIdx = -1;

    // A: toolName{ or toolName({ where toolName is an allowed tool.
    //    Works for both complete and incomplete JSON.
    if (allowedToolNames && allowedToolNames.size > 0) {
      const toolBraceRe = /(\w+)\(?\{/g;
      let m;
      while ((m = toolBraceRe.exec(typed.text)) !== null) {
        if (m[1] && allowedToolNames.has(m[1])) {
          stripIdx = m.index;
          break; // earliest match wins
        }
      }
    }

    // B: Standalone JSON with recognizable tool argument keys.
    const standaloneMatch = typed.text.search(
      /\{\{?\s*"?(?:action|name|tool|function|command|file_path|path|query)/,
    );
    if (standaloneMatch >= 0 && (stripIdx < 0 || standaloneMatch < stripIdx)) {
      stripIdx = standaloneMatch;
    }

    if (stripIdx < 0) {
      continue;
    }

    // Look backward from the strip point for a markdown code fence
    // (e.g. ```json\n) that wraps the tool call — strip from the fence
    // so the UI doesn't show a dangling code block.
    const beforeStrip = typed.text.slice(0, stripIdx);
    const fenceMatch = /```(?:\w+)?\s*\n?\s*$/.exec(beforeStrip);
    if (fenceMatch) {
      stripIdx = fenceMatch.index;
    }

    typed.text = typed.text.slice(0, stripIdx).trimEnd();
  }
}

/**
 * Shallow-clone a streaming partial's content blocks so that
 * `stripPartialInlineToolCallText` mutations don't corrupt the
 * stream's internal accumulation buffer. `streamSimple` reuses
 * a single `output` object as both `event.partial` (streaming)
 * and `event.message` (done) — mutating `.text` on a partial
 * destroys the text the final extraction needs.
 */
function clonePartialForStripping(partial: unknown): unknown {
  if (!partial || typeof partial !== "object") {
    return partial;
  }
  const msg = partial as { content?: unknown[] };
  if (!Array.isArray(msg.content)) {
    return partial;
  }
  return {
    ...msg,
    content: msg.content.map((block) =>
      block && typeof block === "object" ? { ...block } : block,
    ),
  };
}

function wrapStreamExtractDeepSeekInlineToolCalls(
  stream: ReturnType<typeof streamSimple>,
  allowedToolNames?: Set<string>,
): ReturnType<typeof streamSimple> {
  log.info("DeepSeek inline wrapper: wrapping stream for inline tool call extraction");
  const originalResult = stream.result.bind(stream);
  stream.result = async () => {
    const message = await originalResult();
    log.info(
      `DeepSeek inline wrapper: result() called, message content types=[${Array.isArray((message as { content?: unknown[] })?.content) ? (message as { content: { type?: string }[] }).content.map((b) => b?.type).join(",") : "?"}]`,
    );
    extractInlineToolCallsFromMessage(message, allowedToolNames);
    return message;
  };

  const originalAsyncIterator = stream[Symbol.asyncIterator].bind(stream);
  (stream as { [Symbol.asyncIterator]: typeof originalAsyncIterator })[Symbol.asyncIterator] =
    function () {
      const iterator = originalAsyncIterator();
      return {
        async next() {
          const result = await iterator.next();
          if (!result.done && result.value && typeof result.value === "object") {
            const event = result.value as { partial?: unknown; message?: unknown };
            // Extract structured tool calls from the final complete message.
            extractInlineToolCallsFromMessage(event.message, allowedToolNames);
            // Strip tool call text from partials so the UI doesn't render it.
            // Clone the partial first — streamSimple reuses a single object
            // for both partial and done events, so mutating partial text would
            // corrupt the accumulation buffer and prevent final extraction.
            if (event.partial) {
              const cloned = clonePartialForStripping(event.partial);
              stripPartialInlineToolCallText(cloned, allowedToolNames);
              event.partial = cloned;
            }
          }
          return result;
        },
        async return(value?: unknown) {
          return iterator.return?.(value) ?? { done: true as const, value: undefined };
        },
        async throw(error?: unknown) {
          return iterator.throw?.(error) ?? { done: true as const, value: undefined };
        },
      };
    };
  return stream;
}

export function wrapStreamFnExtractDeepSeekInlineToolCalls(
  baseFn: StreamFn,
  allowedToolNames?: Set<string>,
): StreamFn {
  return (model, context, options) => {
    const maybeStream = baseFn(model, context, options);
    if (maybeStream && typeof maybeStream === "object" && "then" in maybeStream) {
      return Promise.resolve(maybeStream).then((stream) =>
        wrapStreamExtractDeepSeekInlineToolCalls(stream, allowedToolNames),
      );
    }
    return wrapStreamExtractDeepSeekInlineToolCalls(maybeStream, allowedToolNames);
  };
}

export async function resolvePromptBuildHookResult(params: {
  prompt: string;
  messages: unknown[];
  hookCtx: PluginHookAgentContext;
  hookRunner?: PromptBuildHookRunner | null;
  legacyBeforeAgentStartResult?: PluginHookBeforeAgentStartResult;
}): Promise<PluginHookBeforePromptBuildResult> {
  const promptBuildResult = params.hookRunner?.hasHooks("before_prompt_build")
    ? await params.hookRunner
        .runBeforePromptBuild(
          {
            prompt: params.prompt,
            messages: params.messages,
          },
          params.hookCtx,
        )
        .catch((hookErr: unknown) => {
          log.warn(`before_prompt_build hook failed: ${String(hookErr)}`);
          return undefined;
        })
    : undefined;
  const legacyResult =
    params.legacyBeforeAgentStartResult ??
    (params.hookRunner?.hasHooks("before_agent_start")
      ? await params.hookRunner
          .runBeforeAgentStart(
            {
              prompt: params.prompt,
              messages: params.messages,
            },
            params.hookCtx,
          )
          .catch((hookErr: unknown) => {
            log.warn(
              `before_agent_start hook (legacy prompt build path) failed: ${String(hookErr)}`,
            );
            return undefined;
          })
      : undefined);
  return {
    systemPrompt: promptBuildResult?.systemPrompt ?? legacyResult?.systemPrompt,
    prependContext: joinPresentTextSegments([
      promptBuildResult?.prependContext,
      legacyResult?.prependContext,
    ]),
    prependSystemContext: joinPresentTextSegments([
      promptBuildResult?.prependSystemContext,
      legacyResult?.prependSystemContext,
    ]),
    appendSystemContext: joinPresentTextSegments([
      promptBuildResult?.appendSystemContext,
      legacyResult?.appendSystemContext,
    ]),
  };
}

export function composeSystemPromptWithHookContext(params: {
  baseSystemPrompt?: string;
  prependSystemContext?: string;
  appendSystemContext?: string;
}): string | undefined {
  const prependSystem = params.prependSystemContext?.trim();
  const appendSystem = params.appendSystemContext?.trim();
  if (!prependSystem && !appendSystem) {
    return undefined;
  }
  return joinPresentTextSegments(
    [params.prependSystemContext, params.baseSystemPrompt, params.appendSystemContext],
    { trim: true },
  );
}

export function resolvePromptModeForSession(sessionKey?: string): "minimal" | "full" {
  if (!sessionKey) {
    return "full";
  }
  return isSubagentSessionKey(sessionKey) || isCronSessionKey(sessionKey) ? "minimal" : "full";
}

export function resolveAttemptFsWorkspaceOnly(params: {
  config?: OpenClawConfig;
  sessionAgentId: string;
}): boolean {
  return resolveEffectiveToolFsWorkspaceOnly({
    cfg: params.config,
    agentId: params.sessionAgentId,
  });
}

export function prependSystemPromptAddition(params: {
  systemPrompt: string;
  systemPromptAddition?: string;
}): string {
  if (!params.systemPromptAddition) {
    return params.systemPrompt;
  }
  return `${params.systemPromptAddition}\n\n${params.systemPrompt}`;
}

/** Build runtime context passed into context-engine afterTurn hooks. */
export function buildAfterTurnRuntimeContext(params: {
  attempt: Pick<
    EmbeddedRunAttemptParams,
    | "sessionKey"
    | "messageChannel"
    | "messageProvider"
    | "agentAccountId"
    | "config"
    | "skillsSnapshot"
    | "senderIsOwner"
    | "provider"
    | "modelId"
    | "thinkLevel"
    | "reasoningLevel"
    | "bashElevated"
    | "extraSystemPrompt"
    | "ownerNumbers"
    | "authProfileId"
  >;
  workspaceDir: string;
  agentDir: string;
}): Partial<CompactEmbeddedPiSessionParams> {
  return {
    sessionKey: params.attempt.sessionKey,
    messageChannel: params.attempt.messageChannel,
    messageProvider: params.attempt.messageProvider,
    agentAccountId: params.attempt.agentAccountId,
    authProfileId: params.attempt.authProfileId,
    workspaceDir: params.workspaceDir,
    agentDir: params.agentDir,
    config: params.attempt.config,
    skillsSnapshot: params.attempt.skillsSnapshot,
    senderIsOwner: params.attempt.senderIsOwner,
    provider: params.attempt.provider,
    model: params.attempt.modelId,
    thinkLevel: params.attempt.thinkLevel,
    reasoningLevel: params.attempt.reasoningLevel,
    bashElevated: params.attempt.bashElevated,
    extraSystemPrompt: params.attempt.extraSystemPrompt,
    ownerNumbers: params.attempt.ownerNumbers,
  };
}

function summarizeMessagePayload(msg: AgentMessage): { textChars: number; imageBlocks: number } {
  const content = (msg as { content?: unknown }).content;
  if (typeof content === "string") {
    return { textChars: content.length, imageBlocks: 0 };
  }
  if (!Array.isArray(content)) {
    return { textChars: 0, imageBlocks: 0 };
  }

  let textChars = 0;
  let imageBlocks = 0;
  for (const block of content) {
    if (!block || typeof block !== "object") {
      continue;
    }
    const typedBlock = block as { type?: unknown; text?: unknown };
    if (typedBlock.type === "image") {
      imageBlocks++;
      continue;
    }
    if (typeof typedBlock.text === "string") {
      textChars += typedBlock.text.length;
    }
  }

  return { textChars, imageBlocks };
}

function summarizeSessionContext(messages: AgentMessage[]): {
  roleCounts: string;
  totalTextChars: number;
  totalImageBlocks: number;
  maxMessageTextChars: number;
} {
  const roleCounts = new Map<string, number>();
  let totalTextChars = 0;
  let totalImageBlocks = 0;
  let maxMessageTextChars = 0;

  for (const msg of messages) {
    const role = typeof msg.role === "string" ? msg.role : "unknown";
    roleCounts.set(role, (roleCounts.get(role) ?? 0) + 1);

    const payload = summarizeMessagePayload(msg);
    totalTextChars += payload.textChars;
    totalImageBlocks += payload.imageBlocks;
    if (payload.textChars > maxMessageTextChars) {
      maxMessageTextChars = payload.textChars;
    }
  }

  return {
    roleCounts:
      [...roleCounts.entries()]
        .toSorted((a, b) => a[0].localeCompare(b[0]))
        .map(([role, count]) => `${role}:${count}`)
        .join(",") || "none",
    totalTextChars,
    totalImageBlocks,
    maxMessageTextChars,
  };
}

export async function runEmbeddedAttempt(
  params: EmbeddedRunAttemptParams,
): Promise<EmbeddedRunAttemptResult> {
  const resolvedWorkspace = resolveUserPath(params.workspaceDir);
  const prevCwd = process.cwd();
  const runAbortController = new AbortController();
  // Proxy bootstrap must happen before timeout tuning so the timeouts wrap the
  // active EnvHttpProxyAgent instead of being replaced by a bare proxy dispatcher.
  ensureGlobalUndiciEnvProxyDispatcher();
  ensureGlobalUndiciStreamTimeouts();

  log.debug(
    `embedded run start: runId=${params.runId} sessionId=${params.sessionId} provider=${params.provider} model=${params.modelId} thinking=${params.thinkLevel} messageChannel=${params.messageChannel ?? params.messageProvider ?? "unknown"}`,
  );

  await fs.mkdir(resolvedWorkspace, { recursive: true });

  const sandboxSessionKey = params.sessionKey?.trim() || params.sessionId;
  const sandbox = await resolveSandboxContext({
    config: params.config,
    sessionKey: sandboxSessionKey,
    workspaceDir: resolvedWorkspace,
  });
  const effectiveWorkspace = sandbox?.enabled
    ? sandbox.workspaceAccess === "rw"
      ? resolvedWorkspace
      : sandbox.workspaceDir
    : resolvedWorkspace;
  await fs.mkdir(effectiveWorkspace, { recursive: true });

  let restoreSkillEnv: (() => void) | undefined;
  process.chdir(effectiveWorkspace);
  try {
    const { shouldLoadSkillEntries, skillEntries } = resolveEmbeddedRunSkillEntries({
      workspaceDir: effectiveWorkspace,
      config: params.config,
      skillsSnapshot: params.skillsSnapshot,
    });
    restoreSkillEnv = params.skillsSnapshot
      ? applySkillEnvOverridesFromSnapshot({
          snapshot: params.skillsSnapshot,
          config: params.config,
        })
      : applySkillEnvOverrides({
          skills: skillEntries ?? [],
          config: params.config,
        });

    const skillsPrompt = resolveSkillsPromptForRun({
      skillsSnapshot: params.skillsSnapshot,
      entries: shouldLoadSkillEntries ? skillEntries : undefined,
      config: params.config,
      workspaceDir: effectiveWorkspace,
    });

    const sessionLabel = params.sessionKey ?? params.sessionId;
    const { bootstrapFiles: hookAdjustedBootstrapFiles, contextFiles } =
      await resolveBootstrapContextForRun({
        workspaceDir: effectiveWorkspace,
        config: params.config,
        sessionKey: params.sessionKey,
        sessionId: params.sessionId,
        warn: makeBootstrapWarn({ sessionLabel, warn: (message) => log.warn(message) }),
        contextMode: params.bootstrapContextMode,
        runKind: params.bootstrapContextRunKind,
      });
    const bootstrapMaxChars = resolveBootstrapMaxChars(params.config);
    const bootstrapTotalMaxChars = resolveBootstrapTotalMaxChars(params.config);
    const bootstrapAnalysis = analyzeBootstrapBudget({
      files: buildBootstrapInjectionStats({
        bootstrapFiles: hookAdjustedBootstrapFiles,
        injectedFiles: contextFiles,
      }),
      bootstrapMaxChars,
      bootstrapTotalMaxChars,
    });
    const bootstrapPromptWarningMode = resolveBootstrapPromptTruncationWarningMode(params.config);
    const bootstrapPromptWarning = buildBootstrapPromptWarning({
      analysis: bootstrapAnalysis,
      mode: bootstrapPromptWarningMode,
      seenSignatures: params.bootstrapPromptWarningSignaturesSeen,
      previousSignature: params.bootstrapPromptWarningSignature,
    });
    const workspaceNotes = hookAdjustedBootstrapFiles.some(
      (file) => file.name === DEFAULT_BOOTSTRAP_FILENAME && !file.missing,
    )
      ? ["Reminder: commit your changes in this workspace after edits."]
      : undefined;

    const agentDir = params.agentDir ?? resolveOpenClawAgentDir();

    const { defaultAgentId, sessionAgentId } = resolveSessionAgentIds({
      sessionKey: params.sessionKey,
      config: params.config,
      agentId: params.agentId,
    });
    const effectiveFsWorkspaceOnly = resolveAttemptFsWorkspaceOnly({
      config: params.config,
      sessionAgentId,
    });
    // Check if the model supports native image input
    const modelHasVision = params.model.input?.includes("image") ?? false;
    const toolsRaw = params.disableTools
      ? []
      : createOpenClawCodingTools({
          agentId: sessionAgentId,
          exec: {
            ...params.execOverrides,
            elevated: params.bashElevated,
          },
          sandbox,
          messageProvider: params.messageChannel ?? params.messageProvider,
          agentAccountId: params.agentAccountId,
          messageTo: params.messageTo,
          messageThreadId: params.messageThreadId,
          groupId: params.groupId,
          groupChannel: params.groupChannel,
          groupSpace: params.groupSpace,
          spawnedBy: params.spawnedBy,
          senderId: params.senderId,
          senderName: params.senderName,
          senderUsername: params.senderUsername,
          senderE164: params.senderE164,
          senderIsOwner: params.senderIsOwner,
          sessionKey: sandboxSessionKey,
          sessionId: params.sessionId,
          runId: params.runId,
          agentDir,
          workspaceDir: effectiveWorkspace,
          config: params.config,
          abortSignal: runAbortController.signal,
          modelProvider: params.model.provider,
          modelId: params.modelId,
          modelContextWindowTokens: params.model.contextWindow,
          modelAuthMode: resolveModelAuthMode(params.model.provider, params.config),
          currentChannelId: params.currentChannelId,
          currentThreadTs: params.currentThreadTs,
          currentMessageId: params.currentMessageId,
          replyToMode: params.replyToMode,
          hasRepliedRef: params.hasRepliedRef,
          modelHasVision,
          requireExplicitMessageTarget:
            params.requireExplicitMessageTarget ?? isSubagentSessionKey(params.sessionKey),
          disableMessageTool: params.disableMessageTool,
        });
    const toolsEnabled = supportsModelTools(params.model);
    const tools = sanitizeToolsForGoogle({
      tools: toolsEnabled ? toolsRaw : [],
      provider: params.provider,
    });
    const clientTools = toolsEnabled ? params.clientTools : undefined;
    const allowedToolNames = collectAllowedToolNames({
      tools,
      clientTools,
    });
    logToolSchemasForGoogle({ tools, provider: params.provider });

    const machineName = await getMachineDisplayName();
    const runtimeChannel = normalizeMessageChannel(params.messageChannel ?? params.messageProvider);
    let runtimeCapabilities = runtimeChannel
      ? (resolveChannelCapabilities({
          cfg: params.config,
          channel: runtimeChannel,
          accountId: params.agentAccountId,
        }) ?? [])
      : undefined;
    if (runtimeChannel === "telegram" && params.config) {
      const inlineButtonsScope = resolveTelegramInlineButtonsScope({
        cfg: params.config,
        accountId: params.agentAccountId ?? undefined,
      });
      if (inlineButtonsScope !== "off") {
        if (!runtimeCapabilities) {
          runtimeCapabilities = [];
        }
        if (
          !runtimeCapabilities.some((cap) => String(cap).trim().toLowerCase() === "inlinebuttons")
        ) {
          runtimeCapabilities.push("inlineButtons");
        }
      }
    }
    const reactionGuidance =
      runtimeChannel && params.config
        ? (() => {
            if (runtimeChannel === "telegram") {
              const resolved = resolveTelegramReactionLevel({
                cfg: params.config,
                accountId: params.agentAccountId ?? undefined,
              });
              const level = resolved.agentReactionGuidance;
              return level ? { level, channel: "Telegram" } : undefined;
            }
            if (runtimeChannel === "signal") {
              const resolved = resolveSignalReactionLevel({
                cfg: params.config,
                accountId: params.agentAccountId ?? undefined,
              });
              const level = resolved.agentReactionGuidance;
              return level ? { level, channel: "Signal" } : undefined;
            }
            return undefined;
          })()
        : undefined;
    const sandboxInfo = buildEmbeddedSandboxInfo(sandbox, params.bashElevated);
    const reasoningTagHint = isReasoningTagProvider(params.provider);
    // Resolve channel-specific message actions for system prompt
    const channelActions = runtimeChannel
      ? listChannelSupportedActions({
          cfg: params.config,
          channel: runtimeChannel,
        })
      : undefined;
    const messageToolHints = runtimeChannel
      ? resolveChannelMessageToolHints({
          cfg: params.config,
          channel: runtimeChannel,
          accountId: params.agentAccountId,
        })
      : undefined;

    const defaultModelRef = resolveDefaultModelForAgent({
      cfg: params.config ?? {},
      agentId: sessionAgentId,
    });
    const defaultModelLabel = `${defaultModelRef.provider}/${defaultModelRef.model}`;
    const { runtimeInfo, userTimezone, userTime, userTimeFormat } = buildSystemPromptParams({
      config: params.config,
      agentId: sessionAgentId,
      workspaceDir: effectiveWorkspace,
      cwd: process.cwd(),
      runtime: {
        host: machineName,
        os: `${os.type()} ${os.release()}`,
        arch: os.arch(),
        node: process.version,
        model: `${params.provider}/${params.modelId}`,
        defaultModel: defaultModelLabel,
        shell: detectRuntimeShell(),
        channel: runtimeChannel,
        capabilities: runtimeCapabilities,
        channelActions,
      },
    });
    const isDefaultAgent = sessionAgentId === defaultAgentId;
    const promptMode = resolvePromptModeForSession(params.sessionKey);
    const docsPath = await resolveOpenClawDocsPath({
      workspaceDir: effectiveWorkspace,
      argv1: process.argv[1],
      cwd: process.cwd(),
      moduleUrl: import.meta.url,
    });
    const ttsHint = params.config ? buildTtsSystemPromptHint(params.config) : undefined;
    const ownerDisplay = resolveOwnerDisplaySetting(params.config);

    const appendPrompt = buildEmbeddedSystemPrompt({
      workspaceDir: effectiveWorkspace,
      defaultThinkLevel: params.thinkLevel,
      reasoningLevel: params.reasoningLevel ?? "off",
      extraSystemPrompt: params.extraSystemPrompt,
      ownerNumbers: params.ownerNumbers,
      ownerDisplay: ownerDisplay.ownerDisplay,
      ownerDisplaySecret: ownerDisplay.ownerDisplaySecret,
      reasoningTagHint,
      heartbeatPrompt: isDefaultAgent
        ? resolveHeartbeatPrompt(params.config?.agents?.defaults?.heartbeat?.prompt)
        : undefined,
      skillsPrompt,
      docsPath: docsPath ?? undefined,
      ttsHint,
      workspaceNotes,
      reactionGuidance,
      promptMode,
      acpEnabled: params.config?.acp?.enabled !== false,
      runtimeInfo,
      messageToolHints,
      sandboxInfo,
      tools,
      modelAliasLines: buildModelAliasLines(params.config),
      userTimezone,
      userTime,
      userTimeFormat,
      contextFiles,
      bootstrapTruncationWarningLines: bootstrapPromptWarning.lines,
      memoryCitationsMode: params.config?.memory?.citations,
    });
    const systemPromptReport = buildSystemPromptReport({
      source: "run",
      generatedAt: Date.now(),
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      provider: params.provider,
      model: params.modelId,
      workspaceDir: effectiveWorkspace,
      bootstrapMaxChars,
      bootstrapTotalMaxChars,
      bootstrapTruncation: buildBootstrapTruncationReportMeta({
        analysis: bootstrapAnalysis,
        warningMode: bootstrapPromptWarningMode,
        warning: bootstrapPromptWarning,
      }),
      sandbox: (() => {
        const runtime = resolveSandboxRuntimeStatus({
          cfg: params.config,
          sessionKey: sandboxSessionKey,
        });
        return { mode: runtime.mode, sandboxed: runtime.sandboxed };
      })(),
      systemPrompt: appendPrompt,
      bootstrapFiles: hookAdjustedBootstrapFiles,
      injectedFiles: contextFiles,
      skillsPrompt,
      tools,
    });
    const systemPromptOverride = createSystemPromptOverride(appendPrompt);
    let systemPromptText = systemPromptOverride();

    const sessionLock = await acquireSessionWriteLock({
      sessionFile: params.sessionFile,
      maxHoldMs: resolveSessionLockMaxHoldFromTimeout({
        timeoutMs: params.timeoutMs,
      }),
    });

    let sessionManager: ReturnType<typeof guardSessionManager> | undefined;
    let session: Awaited<ReturnType<typeof createAgentSession>>["session"] | undefined;
    let removeToolResultContextGuard: (() => void) | undefined;
    try {
      await repairSessionFileIfNeeded({
        sessionFile: params.sessionFile,
        warn: (message) => log.warn(message),
      });
      const hadSessionFile = await fs
        .stat(params.sessionFile)
        .then(() => true)
        .catch(() => false);

      const transcriptPolicy = resolveTranscriptPolicy({
        modelApi: params.model?.api,
        provider: params.provider,
        modelId: params.modelId,
      });

      await prewarmSessionFile(params.sessionFile);
      sessionManager = guardSessionManager(SessionManager.open(params.sessionFile), {
        agentId: sessionAgentId,
        sessionKey: params.sessionKey,
        inputProvenance: params.inputProvenance,
        allowSyntheticToolResults: transcriptPolicy.allowSyntheticToolResults,
        allowedToolNames,
      });
      trackSessionManagerAccess(params.sessionFile);

      if (hadSessionFile && params.contextEngine?.bootstrap) {
        try {
          await params.contextEngine.bootstrap({
            sessionId: params.sessionId,
            sessionFile: params.sessionFile,
          });
        } catch (bootstrapErr) {
          log.warn(`context engine bootstrap failed: ${String(bootstrapErr)}`);
        }
      }

      await prepareSessionManagerForRun({
        sessionManager,
        sessionFile: params.sessionFile,
        hadSessionFile,
        sessionId: params.sessionId,
        cwd: effectiveWorkspace,
      });

      const settingsManager = createPreparedEmbeddedPiSettingsManager({
        cwd: effectiveWorkspace,
        agentDir,
        cfg: params.config,
      });
      applyPiAutoCompactionGuard({
        settingsManager,
        contextEngineInfo: params.contextEngine?.info,
      });

      // Sets compaction/pruning runtime state and returns extension factories
      // that must be passed to the resource loader for the safeguard to be active.
      const extensionFactories = buildEmbeddedExtensionFactories({
        cfg: params.config,
        sessionManager,
        provider: params.provider,
        modelId: params.modelId,
        model: params.model,
      });
      // Only create an explicit resource loader when there are extension factories
      // to register; otherwise let createAgentSession use its built-in default.
      let resourceLoader: DefaultResourceLoader | undefined;
      if (extensionFactories.length > 0) {
        resourceLoader = new DefaultResourceLoader({
          cwd: resolvedWorkspace,
          agentDir,
          settingsManager,
          extensionFactories,
        });
        await resourceLoader.reload();
      }

      // Get hook runner early so it's available when creating tools
      const hookRunner = getGlobalHookRunner();

      const { builtInTools, customTools } = splitSdkTools({
        tools,
        sandboxEnabled: !!sandbox?.enabled,
      });

      // Add client tools (OpenResponses hosted tools) to customTools
      let clientToolCallDetected: { name: string; params: Record<string, unknown> } | null = null;
      const clientToolLoopDetection = resolveToolLoopDetectionConfig({
        cfg: params.config,
        agentId: sessionAgentId,
      });
      const clientToolDefs = clientTools
        ? toClientToolDefinitions(
            clientTools,
            (toolName, toolParams) => {
              clientToolCallDetected = { name: toolName, params: toolParams };
            },
            {
              agentId: sessionAgentId,
              sessionKey: sandboxSessionKey,
              sessionId: params.sessionId,
              runId: params.runId,
              loopDetection: clientToolLoopDetection,
            },
          )
        : [];

      const allCustomTools = [...customTools, ...clientToolDefs];

      ({ session } = await createAgentSession({
        cwd: resolvedWorkspace,
        agentDir,
        authStorage: params.authStorage,
        modelRegistry: params.modelRegistry,
        model: params.model,
        thinkingLevel: mapThinkingLevel(params.thinkLevel),
        tools: builtInTools,
        customTools: allCustomTools,
        sessionManager,
        settingsManager,
        resourceLoader,
      }));
      applySystemPromptOverrideToSession(session, systemPromptText);
      if (!session) {
        throw new Error("Embedded agent session missing");
      }
      const activeSession = session;
      removeToolResultContextGuard = installToolResultContextGuard({
        agent: activeSession.agent,
        contextWindowTokens: Math.max(
          1,
          Math.floor(
            params.model.contextWindow ?? params.model.maxTokens ?? DEFAULT_CONTEXT_TOKENS,
          ),
        ),
      });
      const cacheTrace = createCacheTrace({
        cfg: params.config,
        env: process.env,
        runId: params.runId,
        sessionId: activeSession.sessionId,
        sessionKey: params.sessionKey,
        provider: params.provider,
        modelId: params.modelId,
        modelApi: params.model.api,
        workspaceDir: params.workspaceDir,
      });
      const anthropicPayloadLogger = createAnthropicPayloadLogger({
        env: process.env,
        runId: params.runId,
        sessionId: activeSession.sessionId,
        sessionKey: params.sessionKey,
        provider: params.provider,
        modelId: params.modelId,
        modelApi: params.model.api,
        workspaceDir: params.workspaceDir,
      });

      // Ollama native API: bypass SDK's streamSimple and use direct /api/chat calls
      // for reliable streaming + tool calling support (#11828).
      if (params.model.api === "ollama") {
        // Prioritize configured provider baseUrl so Docker/remote Ollama hosts work reliably.
        const providerConfig = params.config?.models?.providers?.[params.model.provider];
        const providerBaseUrl =
          typeof providerConfig?.baseUrl === "string" ? providerConfig.baseUrl : undefined;
        const ollamaStreamFn = createConfiguredOllamaStreamFn({
          model: params.model,
          providerBaseUrl,
        });
        activeSession.agent.streamFn = ollamaStreamFn;
        ensureCustomApiRegistered(params.model.api, ollamaStreamFn);
      } else if (params.model.api === "openai-responses" && params.provider === "openai") {
        const wsApiKey = await params.authStorage.getApiKey(params.provider);
        if (wsApiKey) {
          activeSession.agent.streamFn = createOpenAIWebSocketStreamFn(wsApiKey, params.sessionId, {
            signal: runAbortController.signal,
          });
        } else {
          log.warn(`[ws-stream] no API key for provider=${params.provider}; using HTTP transport`);
          activeSession.agent.streamFn = streamSimple;
        }
      } else {
        // Force a stable streamFn reference so vitest can reliably mock @mariozechner/pi-ai.
        activeSession.agent.streamFn = streamSimple;
      }

      // Ollama with OpenAI-compatible API needs num_ctx in payload.options.
      // Otherwise Ollama defaults to a 4096 context window.
      const providerIdForNumCtx =
        typeof params.model.provider === "string" && params.model.provider.trim().length > 0
          ? params.model.provider
          : params.provider;
      const shouldInjectNumCtx = shouldInjectOllamaCompatNumCtx({
        model: params.model,
        config: params.config,
        providerId: providerIdForNumCtx,
      });
      if (shouldInjectNumCtx) {
        const numCtx = Math.max(
          1,
          Math.floor(
            params.model.contextWindow ?? params.model.maxTokens ?? DEFAULT_CONTEXT_TOKENS,
          ),
        );
        activeSession.agent.streamFn = wrapOllamaCompatNumCtx(activeSession.agent.streamFn, numCtx);
      }

      applyExtraParamsToAgent(
        activeSession.agent,
        params.config,
        params.provider,
        params.modelId,
        params.streamParams,
        params.thinkLevel,
        sessionAgentId,
      );

      if (cacheTrace) {
        cacheTrace.recordStage("session:loaded", {
          messages: activeSession.messages,
          system: systemPromptText,
          note: "after session create",
        });
        activeSession.agent.streamFn = cacheTrace.wrapStreamFn(activeSession.agent.streamFn);
      }

      // Copilot/Claude can reject persisted `thinking` blocks (e.g. thinkingSignature:"reasoning_text")
      // on *any* follow-up provider call (including tool continuations). Wrap the stream function
      // so every outbound request sees sanitized messages.
      if (transcriptPolicy.dropThinkingBlocks) {
        const inner = activeSession.agent.streamFn;
        activeSession.agent.streamFn = (model, context, options) => {
          const ctx = context as unknown as { messages?: unknown };
          const messages = ctx?.messages;
          if (!Array.isArray(messages)) {
            return inner(model, context, options);
          }
          const sanitized = dropThinkingBlocks(messages as unknown as AgentMessage[]) as unknown;
          if (sanitized === messages) {
            return inner(model, context, options);
          }
          const nextContext = {
            ...(context as unknown as Record<string, unknown>),
            messages: sanitized,
          } as unknown;
          return inner(model, nextContext as typeof context, options);
        };
      }

      // Mistral (and other strict providers) reject tool call IDs that don't match their
      // format requirements (e.g. [a-zA-Z0-9]{9}). sanitizeSessionHistory only processes
      // historical messages at attempt start, but the agent loop's internal tool call →
      // tool result cycles bypass that path. Wrap streamFn so every outbound request
      // sees sanitized tool call IDs.
      if (transcriptPolicy.sanitizeToolCallIds && transcriptPolicy.toolCallIdMode) {
        const inner = activeSession.agent.streamFn;
        const mode = transcriptPolicy.toolCallIdMode;
        activeSession.agent.streamFn = (model, context, options) => {
          const ctx = context as unknown as { messages?: unknown };
          const messages = ctx?.messages;
          if (!Array.isArray(messages)) {
            return inner(model, context, options);
          }
          const sanitized = sanitizeToolCallIdsForCloudCodeAssist(messages as AgentMessage[], mode);
          if (sanitized === messages) {
            return inner(model, context, options);
          }
          const nextContext = {
            ...(context as unknown as Record<string, unknown>),
            messages: sanitized,
          } as unknown;
          return inner(model, nextContext as typeof context, options);
        };
      }

      if (
        params.model.api === "openai-responses" ||
        params.model.api === "openai-codex-responses"
      ) {
        const inner = activeSession.agent.streamFn;
        activeSession.agent.streamFn = (model, context, options) => {
          const ctx = context as unknown as { messages?: unknown };
          const messages = ctx?.messages;
          if (!Array.isArray(messages)) {
            return inner(model, context, options);
          }
          const sanitized = downgradeOpenAIFunctionCallReasoningPairs(messages as AgentMessage[]);
          if (sanitized === messages) {
            return inner(model, context, options);
          }
          const nextContext = {
            ...(context as unknown as Record<string, unknown>),
            messages: sanitized,
          } as unknown;
          return inner(model, nextContext as typeof context, options);
        };
      }

      // Some models emit tool names with surrounding whitespace (e.g. " read ").
      // pi-agent-core dispatches tool calls with exact string matching, so normalize
      // names on the live response stream before tool execution.
      activeSession.agent.streamFn = wrapStreamFnTrimToolCallNames(
        activeSession.agent.streamFn,
        allowedToolNames,
      );

      if (isXaiProvider(params.provider, params.modelId)) {
        activeSession.agent.streamFn = wrapStreamFnDecodeXaiToolCallArguments(
          activeSession.agent.streamFn,
        );
      }

      // DeepSeek on Azure AI Foundry sometimes emits tool calls as inline text
      // (e.g. `exec{"command":"..."}`) instead of structured tool_calls.
      // Extract them into proper tool call blocks so pi-agent-core can dispatch,
      // and strip the raw text from streaming partials so the UI doesn't show it.
      const deepSeekAzureMatch = isDeepSeekOnAzureFoundry(params.modelId, params.model.baseUrl);
      log.info(
        `DeepSeek inline extraction check: modelId=${params.modelId} baseUrl=${params.model.baseUrl} match=${deepSeekAzureMatch} allowedTools=[${[...allowedToolNames].join(",")}]`,
      );
      if (deepSeekAzureMatch) {
        activeSession.agent.streamFn = wrapStreamFnExtractDeepSeekInlineToolCalls(
          activeSession.agent.streamFn,
          allowedToolNames,
        );
      }

      if (anthropicPayloadLogger) {
        activeSession.agent.streamFn = anthropicPayloadLogger.wrapStreamFn(
          activeSession.agent.streamFn,
        );
      }

      try {
        const prior = await sanitizeSessionHistory({
          messages: activeSession.messages,
          modelApi: params.model.api,
          modelId: params.modelId,
          provider: params.provider,
          allowedToolNames,
          config: params.config,
          sessionManager,
          sessionId: params.sessionId,
          policy: transcriptPolicy,
        });
        cacheTrace?.recordStage("session:sanitized", { messages: prior });
        const validatedGemini = transcriptPolicy.validateGeminiTurns
          ? validateGeminiTurns(prior)
          : prior;
        const validated = transcriptPolicy.validateAnthropicTurns
          ? validateAnthropicTurns(validatedGemini)
          : validatedGemini;
        const truncated = limitHistoryTurns(
          validated,
          getDmHistoryLimitFromSessionKey(params.sessionKey, params.config),
        );
        // Re-run tool_use/tool_result pairing repair after truncation, since
        // limitHistoryTurns can orphan tool_result blocks by removing the
        // assistant message that contained the matching tool_use.
        const limited = transcriptPolicy.repairToolUseResultPairing
          ? sanitizeToolUseResultPairing(truncated)
          : truncated;
        cacheTrace?.recordStage("session:limited", { messages: limited });
        if (limited.length > 0) {
          activeSession.agent.replaceMessages(limited);
        }

        if (params.contextEngine) {
          try {
            const assembled = await params.contextEngine.assemble({
              sessionId: params.sessionId,
              messages: activeSession.messages,
              tokenBudget: params.contextTokenBudget,
            });
            if (assembled.messages !== activeSession.messages) {
              activeSession.agent.replaceMessages(assembled.messages);
            }
            if (assembled.systemPromptAddition) {
              systemPromptText = prependSystemPromptAddition({
                systemPrompt: systemPromptText,
                systemPromptAddition: assembled.systemPromptAddition,
              });
              applySystemPromptOverrideToSession(activeSession, systemPromptText);
              log.debug(
                `context engine: prepended system prompt addition (${assembled.systemPromptAddition.length} chars)`,
              );
            }
          } catch (assembleErr) {
            log.warn(
              `context engine assemble failed, using pipeline messages: ${String(assembleErr)}`,
            );
          }
        }
      } catch (err) {
        await flushPendingToolResultsAfterIdle({
          agent: activeSession?.agent,
          sessionManager,
          clearPendingOnTimeout: true,
        });
        activeSession.dispose();
        throw err;
      }

      let aborted = Boolean(params.abortSignal?.aborted);
      let timedOut = false;
      let timedOutDuringCompaction = false;
      const getAbortReason = (signal: AbortSignal): unknown =>
        "reason" in signal ? (signal as { reason?: unknown }).reason : undefined;
      const makeTimeoutAbortReason = (): Error => {
        const err = new Error("request timed out");
        err.name = "TimeoutError";
        return err;
      };
      const makeAbortError = (signal: AbortSignal): Error => {
        const reason = getAbortReason(signal);
        const err = reason ? new Error("aborted", { cause: reason }) : new Error("aborted");
        err.name = "AbortError";
        return err;
      };
      const abortRun = (isTimeout = false, reason?: unknown) => {
        aborted = true;
        if (isTimeout) {
          timedOut = true;
        }
        if (isTimeout) {
          runAbortController.abort(reason ?? makeTimeoutAbortReason());
        } else {
          runAbortController.abort(reason);
        }
        void activeSession.abort();
      };
      const abortable = <T>(promise: Promise<T>): Promise<T> => {
        const signal = runAbortController.signal;
        if (signal.aborted) {
          return Promise.reject(makeAbortError(signal));
        }
        return new Promise<T>((resolve, reject) => {
          const onAbort = () => {
            signal.removeEventListener("abort", onAbort);
            reject(makeAbortError(signal));
          };
          signal.addEventListener("abort", onAbort, { once: true });
          promise.then(
            (value) => {
              signal.removeEventListener("abort", onAbort);
              resolve(value);
            },
            (err) => {
              signal.removeEventListener("abort", onAbort);
              reject(err);
            },
          );
        });
      };

      const subscription = subscribeEmbeddedPiSession({
        session: activeSession,
        runId: params.runId,
        hookRunner: getGlobalHookRunner() ?? undefined,
        verboseLevel: params.verboseLevel,
        reasoningMode: params.reasoningLevel ?? "off",
        toolResultFormat: params.toolResultFormat,
        shouldEmitToolResult: params.shouldEmitToolResult,
        shouldEmitToolOutput: params.shouldEmitToolOutput,
        onToolResult: params.onToolResult,
        onReasoningStream: params.onReasoningStream,
        onReasoningEnd: params.onReasoningEnd,
        onBlockReply: params.onBlockReply,
        onBlockReplyFlush: params.onBlockReplyFlush,
        blockReplyBreak: params.blockReplyBreak,
        blockReplyChunking: params.blockReplyChunking,
        onPartialReply: params.onPartialReply,
        onAssistantMessageStart: params.onAssistantMessageStart,
        onAgentEvent: params.onAgentEvent,
        enforceFinalTag: params.enforceFinalTag,
        config: params.config,
        sessionKey: sandboxSessionKey,
        sessionId: params.sessionId,
        agentId: sessionAgentId,
      });

      const {
        assistantTexts,
        toolMetas,
        unsubscribe,
        waitForCompactionRetry,
        isCompactionInFlight,
        getMessagingToolSentTexts,
        getMessagingToolSentMediaUrls,
        getMessagingToolSentTargets,
        getSuccessfulCronAdds,
        didSendViaMessagingTool,
        getLastToolError,
        getUsageTotals,
        getCompactionCount,
      } = subscription;

      const queueHandle: EmbeddedPiQueueHandle = {
        queueMessage: async (text: string) => {
          await activeSession.steer(text);
        },
        isStreaming: () => activeSession.isStreaming,
        isCompacting: () => subscription.isCompacting(),
        abort: abortRun,
      };
      setActiveEmbeddedRun(params.sessionId, queueHandle, params.sessionKey);

      let abortWarnTimer: NodeJS.Timeout | undefined;
      const isProbeSession = params.sessionId?.startsWith("probe-") ?? false;
      const abortTimer = setTimeout(
        () => {
          if (!isProbeSession) {
            log.warn(
              `embedded run timeout: runId=${params.runId} sessionId=${params.sessionId} timeoutMs=${params.timeoutMs}`,
            );
          }
          if (
            shouldFlagCompactionTimeout({
              isTimeout: true,
              isCompactionPendingOrRetrying: subscription.isCompacting(),
              isCompactionInFlight: activeSession.isCompacting,
            })
          ) {
            timedOutDuringCompaction = true;
          }
          abortRun(true);
          if (!abortWarnTimer) {
            abortWarnTimer = setTimeout(() => {
              if (!activeSession.isStreaming) {
                return;
              }
              if (!isProbeSession) {
                log.warn(
                  `embedded run abort still streaming: runId=${params.runId} sessionId=${params.sessionId}`,
                );
              }
            }, 10_000);
          }
        },
        Math.max(1, params.timeoutMs),
      );

      let messagesSnapshot: AgentMessage[] = [];
      let sessionIdUsed = activeSession.sessionId;
      const onAbort = () => {
        const reason = params.abortSignal ? getAbortReason(params.abortSignal) : undefined;
        const timeout = reason ? isTimeoutError(reason) : false;
        if (
          shouldFlagCompactionTimeout({
            isTimeout: timeout,
            isCompactionPendingOrRetrying: subscription.isCompacting(),
            isCompactionInFlight: activeSession.isCompacting,
          })
        ) {
          timedOutDuringCompaction = true;
        }
        abortRun(timeout, reason);
      };
      if (params.abortSignal) {
        if (params.abortSignal.aborted) {
          onAbort();
        } else {
          params.abortSignal.addEventListener("abort", onAbort, {
            once: true,
          });
        }
      }

      // Hook runner was already obtained earlier before tool creation
      const hookAgentId = sessionAgentId;

      let promptError: unknown = null;
      let promptErrorSource: "prompt" | "compaction" | null = null;
      const prePromptMessageCount = activeSession.messages.length;
      try {
        const promptStartedAt = Date.now();

        // Run before_prompt_build hooks to allow plugins to inject prompt context.
        // Legacy compatibility: before_agent_start is also checked for context fields.
        let effectivePrompt = params.prompt;
        const hookCtx = {
          agentId: hookAgentId,
          sessionKey: params.sessionKey,
          sessionId: params.sessionId,
          workspaceDir: params.workspaceDir,
          messageProvider: params.messageProvider ?? undefined,
          trigger: params.trigger,
          channelId: params.messageChannel ?? params.messageProvider ?? undefined,
        };
        const hookResult = await resolvePromptBuildHookResult({
          prompt: params.prompt,
          messages: activeSession.messages,
          hookCtx,
          hookRunner,
          legacyBeforeAgentStartResult: params.legacyBeforeAgentStartResult,
        });
        {
          if (hookResult?.prependContext) {
            effectivePrompt = `${hookResult.prependContext}\n\n${params.prompt}`;
            log.debug(
              `hooks: prepended context to prompt (${hookResult.prependContext.length} chars)`,
            );
          }
          const legacySystemPrompt =
            typeof hookResult?.systemPrompt === "string" ? hookResult.systemPrompt.trim() : "";
          if (legacySystemPrompt) {
            applySystemPromptOverrideToSession(activeSession, legacySystemPrompt);
            systemPromptText = legacySystemPrompt;
            log.debug(`hooks: applied systemPrompt override (${legacySystemPrompt.length} chars)`);
          }
          const prependedOrAppendedSystemPrompt = composeSystemPromptWithHookContext({
            baseSystemPrompt: systemPromptText,
            prependSystemContext: hookResult?.prependSystemContext,
            appendSystemContext: hookResult?.appendSystemContext,
          });
          if (prependedOrAppendedSystemPrompt) {
            const prependSystemLen = hookResult?.prependSystemContext?.trim().length ?? 0;
            const appendSystemLen = hookResult?.appendSystemContext?.trim().length ?? 0;
            applySystemPromptOverrideToSession(activeSession, prependedOrAppendedSystemPrompt);
            systemPromptText = prependedOrAppendedSystemPrompt;
            log.debug(
              `hooks: applied prependSystemContext/appendSystemContext (${prependSystemLen}+${appendSystemLen} chars)`,
            );
          }
        }

        log.debug(`embedded run prompt start: runId=${params.runId} sessionId=${params.sessionId}`);
        cacheTrace?.recordStage("prompt:before", {
          prompt: effectivePrompt,
          messages: activeSession.messages,
        });

        // Repair orphaned trailing user messages so new prompts don't violate role ordering.
        const leafEntry = sessionManager.getLeafEntry();
        if (leafEntry?.type === "message" && leafEntry.message.role === "user") {
          if (leafEntry.parentId) {
            sessionManager.branch(leafEntry.parentId);
          } else {
            sessionManager.resetLeaf();
          }
          const sessionContext = sessionManager.buildSessionContext();
          activeSession.agent.replaceMessages(sessionContext.messages);
          log.warn(
            `Removed orphaned user message to prevent consecutive user turns. ` +
              `runId=${params.runId} sessionId=${params.sessionId}`,
          );
        }

        try {
          // Idempotent cleanup for legacy sessions with persisted image payloads.
          // Called each run; only mutates already-answered user turns that still carry image blocks.
          const didPruneImages = pruneProcessedHistoryImages(activeSession.messages);
          if (didPruneImages) {
            activeSession.agent.replaceMessages(activeSession.messages);
          }

          // Detect and load images referenced in the prompt for vision-capable models.
          // Images are prompt-local only (pi-like behavior).
          const imageResult = await detectAndLoadPromptImages({
            prompt: effectivePrompt,
            workspaceDir: effectiveWorkspace,
            model: params.model,
            existingImages: params.images,
            maxBytes: MAX_IMAGE_BYTES,
            maxDimensionPx: resolveImageSanitizationLimits(params.config).maxDimensionPx,
            workspaceOnly: effectiveFsWorkspaceOnly,
            // Enforce sandbox path restrictions when sandbox is enabled
            sandbox:
              sandbox?.enabled && sandbox?.fsBridge
                ? { root: sandbox.workspaceDir, bridge: sandbox.fsBridge }
                : undefined,
          });

          cacheTrace?.recordStage("prompt:images", {
            prompt: effectivePrompt,
            messages: activeSession.messages,
            note: `images: prompt=${imageResult.images.length}`,
          });

          // Diagnostic: log context sizes before prompt to help debug early overflow errors.
          if (log.isEnabled("debug")) {
            const msgCount = activeSession.messages.length;
            const systemLen = systemPromptText?.length ?? 0;
            const promptLen = effectivePrompt.length;
            const sessionSummary = summarizeSessionContext(activeSession.messages);
            log.debug(
              `[context-diag] pre-prompt: sessionKey=${params.sessionKey ?? params.sessionId} ` +
                `messages=${msgCount} roleCounts=${sessionSummary.roleCounts} ` +
                `historyTextChars=${sessionSummary.totalTextChars} ` +
                `maxMessageTextChars=${sessionSummary.maxMessageTextChars} ` +
                `historyImageBlocks=${sessionSummary.totalImageBlocks} ` +
                `systemPromptChars=${systemLen} promptChars=${promptLen} ` +
                `promptImages=${imageResult.images.length} ` +
                `provider=${params.provider}/${params.modelId} sessionFile=${params.sessionFile}`,
            );
          }

          if (hookRunner?.hasHooks("llm_input")) {
            hookRunner
              .runLlmInput(
                {
                  runId: params.runId,
                  sessionId: params.sessionId,
                  provider: params.provider,
                  model: params.modelId,
                  systemPrompt: systemPromptText,
                  prompt: effectivePrompt,
                  historyMessages: activeSession.messages,
                  imagesCount: imageResult.images.length,
                },
                {
                  agentId: hookAgentId,
                  sessionKey: params.sessionKey,
                  sessionId: params.sessionId,
                  workspaceDir: params.workspaceDir,
                  messageProvider: params.messageProvider ?? undefined,
                  trigger: params.trigger,
                  channelId: params.messageChannel ?? params.messageProvider ?? undefined,
                },
              )
              .catch((err) => {
                log.warn(`llm_input hook failed: ${String(err)}`);
              });
          }

          // Only pass images option if there are actually images to pass
          // This avoids potential issues with models that don't expect the images parameter
          if (imageResult.images.length > 0) {
            await abortable(activeSession.prompt(effectivePrompt, { images: imageResult.images }));
          } else {
            await abortable(activeSession.prompt(effectivePrompt));
          }
        } catch (err) {
          promptError = err;
          promptErrorSource = "prompt";
        } finally {
          log.debug(
            `embedded run prompt end: runId=${params.runId} sessionId=${params.sessionId} durationMs=${Date.now() - promptStartedAt}`,
          );
        }

        // Capture snapshot before compaction wait so we have complete messages if timeout occurs
        // Check compaction state before and after to avoid race condition where compaction starts during capture
        // Use session state (not subscription) for snapshot decisions - need instantaneous compaction status
        const wasCompactingBefore = activeSession.isCompacting;
        const snapshot = activeSession.messages.slice();
        const wasCompactingAfter = activeSession.isCompacting;
        // Only trust snapshot if compaction wasn't running before or after capture
        const preCompactionSnapshot = wasCompactingBefore || wasCompactingAfter ? null : snapshot;
        const preCompactionSessionId = activeSession.sessionId;
        const COMPACTION_RETRY_AGGREGATE_TIMEOUT_MS = 60_000;

        try {
          // Flush buffered block replies before waiting for compaction so the
          // user receives the assistant response immediately.  Without this,
          // coalesced/buffered blocks stay in the pipeline until compaction
          // finishes — which can take minutes on large contexts (#35074).
          if (params.onBlockReplyFlush) {
            await params.onBlockReplyFlush();
          }

          const compactionRetryWait = await waitForCompactionRetryWithAggregateTimeout({
            waitForCompactionRetry,
            abortable,
            aggregateTimeoutMs: COMPACTION_RETRY_AGGREGATE_TIMEOUT_MS,
            isCompactionStillInFlight: isCompactionInFlight,
          });
          if (compactionRetryWait.timedOut) {
            timedOutDuringCompaction = true;
            if (!isProbeSession) {
              log.warn(
                `compaction retry aggregate timeout (${COMPACTION_RETRY_AGGREGATE_TIMEOUT_MS}ms): ` +
                  `proceeding with pre-compaction state runId=${params.runId} sessionId=${params.sessionId}`,
              );
            }
          }
        } catch (err) {
          if (isRunnerAbortError(err)) {
            if (!promptError) {
              promptError = err;
              promptErrorSource = "compaction";
            }
            if (!isProbeSession) {
              log.debug(
                `compaction wait aborted: runId=${params.runId} sessionId=${params.sessionId}`,
              );
            }
          } else {
            throw err;
          }
        }

        const compactionOccurredThisAttempt = getCompactionCount() > 0;

        // Append cache-TTL timestamp AFTER prompt + compaction retry completes.
        // Previously this was before the prompt, which caused a custom entry to be
        // inserted between compaction and the next prompt — breaking the
        // prepareCompaction() guard that checks the last entry type, leading to
        // double-compaction. See: https://github.com/openclaw/openclaw/issues/9282
        // Skip when timed out during compaction — session state may be inconsistent.
        if (!timedOutDuringCompaction && !compactionOccurredThisAttempt) {
          const shouldTrackCacheTtl =
            params.config?.agents?.defaults?.contextPruning?.mode === "cache-ttl" &&
            isCacheTtlEligibleProvider(params.provider, params.modelId);
          if (shouldTrackCacheTtl) {
            appendCacheTtlTimestamp(sessionManager, {
              timestamp: Date.now(),
              provider: params.provider,
              modelId: params.modelId,
            });
          }
        }

        // If timeout occurred during compaction, use pre-compaction snapshot when available
        // (compaction restructures messages but does not add user/assistant turns).
        const snapshotSelection = selectCompactionTimeoutSnapshot({
          timedOutDuringCompaction,
          preCompactionSnapshot,
          preCompactionSessionId,
          currentSnapshot: activeSession.messages.slice(),
          currentSessionId: activeSession.sessionId,
        });
        if (timedOutDuringCompaction) {
          if (!isProbeSession) {
            log.warn(
              `using ${snapshotSelection.source} snapshot: timed out during compaction runId=${params.runId} sessionId=${params.sessionId}`,
            );
          }
        }
        messagesSnapshot = snapshotSelection.messagesSnapshot;
        sessionIdUsed = snapshotSelection.sessionIdUsed;

        if (promptError && promptErrorSource === "prompt" && !compactionOccurredThisAttempt) {
          try {
            sessionManager.appendCustomEntry("openclaw:prompt-error", {
              timestamp: Date.now(),
              runId: params.runId,
              sessionId: params.sessionId,
              provider: params.provider,
              model: params.modelId,
              api: params.model.api,
              error: describeUnknownError(promptError),
            });
          } catch (entryErr) {
            log.warn(`failed to persist prompt error entry: ${String(entryErr)}`);
          }
        }

        // Let the active context engine run its post-turn lifecycle.
        if (params.contextEngine) {
          const afterTurnRuntimeContext = buildAfterTurnRuntimeContext({
            attempt: params,
            workspaceDir: effectiveWorkspace,
            agentDir,
          });

          if (typeof params.contextEngine.afterTurn === "function") {
            try {
              await params.contextEngine.afterTurn({
                sessionId: sessionIdUsed,
                sessionFile: params.sessionFile,
                messages: messagesSnapshot,
                prePromptMessageCount,
                tokenBudget: params.contextTokenBudget,
                runtimeContext: afterTurnRuntimeContext,
              });
            } catch (afterTurnErr) {
              log.warn(`context engine afterTurn failed: ${String(afterTurnErr)}`);
            }
          } else {
            // Fallback: ingest new messages individually
            const newMessages = messagesSnapshot.slice(prePromptMessageCount);
            if (newMessages.length > 0) {
              if (typeof params.contextEngine.ingestBatch === "function") {
                try {
                  await params.contextEngine.ingestBatch({
                    sessionId: sessionIdUsed,
                    messages: newMessages,
                  });
                } catch (ingestErr) {
                  log.warn(`context engine ingest failed: ${String(ingestErr)}`);
                }
              } else {
                for (const msg of newMessages) {
                  try {
                    await params.contextEngine.ingest({
                      sessionId: sessionIdUsed,
                      message: msg,
                    });
                  } catch (ingestErr) {
                    log.warn(`context engine ingest failed: ${String(ingestErr)}`);
                  }
                }
              }
            }
          }
        }

        cacheTrace?.recordStage("session:after", {
          messages: messagesSnapshot,
          note: timedOutDuringCompaction
            ? "compaction timeout"
            : promptError
              ? "prompt error"
              : undefined,
        });
        anthropicPayloadLogger?.recordUsage(messagesSnapshot, promptError);

        // Run agent_end hooks to allow plugins to analyze the conversation
        // This is fire-and-forget, so we don't await
        // Run even on compaction timeout so plugins can log/cleanup
        if (hookRunner?.hasHooks("agent_end")) {
          hookRunner
            .runAgentEnd(
              {
                messages: messagesSnapshot,
                success: !aborted && !promptError,
                error: promptError ? describeUnknownError(promptError) : undefined,
                durationMs: Date.now() - promptStartedAt,
              },
              {
                agentId: hookAgentId,
                sessionKey: params.sessionKey,
                sessionId: params.sessionId,
                workspaceDir: params.workspaceDir,
                messageProvider: params.messageProvider ?? undefined,
                trigger: params.trigger,
                channelId: params.messageChannel ?? params.messageProvider ?? undefined,
              },
            )
            .catch((err) => {
              log.warn(`agent_end hook failed: ${err}`);
            });
        }
      } finally {
        clearTimeout(abortTimer);
        if (abortWarnTimer) {
          clearTimeout(abortWarnTimer);
        }
        if (!isProbeSession && (aborted || timedOut) && !timedOutDuringCompaction) {
          log.debug(
            `run cleanup: runId=${params.runId} sessionId=${params.sessionId} aborted=${aborted} timedOut=${timedOut}`,
          );
        }
        try {
          unsubscribe();
        } catch (err) {
          // unsubscribe() should never throw; if it does, it indicates a serious bug.
          // Log at error level to ensure visibility, but don't rethrow in finally block
          // as it would mask any exception from the try block above.
          log.error(
            `CRITICAL: unsubscribe failed, possible resource leak: runId=${params.runId} ${String(err)}`,
          );
        }
        clearActiveEmbeddedRun(params.sessionId, queueHandle, params.sessionKey);
        params.abortSignal?.removeEventListener?.("abort", onAbort);
      }

      const lastAssistant = messagesSnapshot
        .slice()
        .toReversed()
        .find((m) => m.role === "assistant");

      const toolMetasNormalized = toolMetas
        .filter(
          (entry): entry is { toolName: string; meta?: string } =>
            typeof entry.toolName === "string" && entry.toolName.trim().length > 0,
        )
        .map((entry) => ({ toolName: entry.toolName, meta: entry.meta }));

      if (hookRunner?.hasHooks("llm_output")) {
        hookRunner
          .runLlmOutput(
            {
              runId: params.runId,
              sessionId: params.sessionId,
              provider: params.provider,
              model: params.modelId,
              assistantTexts,
              lastAssistant,
              usage: getUsageTotals(),
            },
            {
              agentId: hookAgentId,
              sessionKey: params.sessionKey,
              sessionId: params.sessionId,
              workspaceDir: params.workspaceDir,
              messageProvider: params.messageProvider ?? undefined,
              trigger: params.trigger,
              channelId: params.messageChannel ?? params.messageProvider ?? undefined,
            },
          )
          .catch((err) => {
            log.warn(`llm_output hook failed: ${String(err)}`);
          });
      }

      return {
        aborted,
        timedOut,
        timedOutDuringCompaction,
        promptError,
        sessionIdUsed,
        bootstrapPromptWarningSignaturesSeen: bootstrapPromptWarning.warningSignaturesSeen,
        bootstrapPromptWarningSignature: bootstrapPromptWarning.signature,
        systemPromptReport,
        messagesSnapshot,
        assistantTexts,
        toolMetas: toolMetasNormalized,
        lastAssistant,
        lastToolError: getLastToolError?.(),
        didSendViaMessagingTool: didSendViaMessagingTool(),
        messagingToolSentTexts: getMessagingToolSentTexts(),
        messagingToolSentMediaUrls: getMessagingToolSentMediaUrls(),
        messagingToolSentTargets: getMessagingToolSentTargets(),
        successfulCronAdds: getSuccessfulCronAdds(),
        cloudCodeAssistFormatError: Boolean(
          lastAssistant?.errorMessage && isCloudCodeAssistFormatError(lastAssistant.errorMessage),
        ),
        attemptUsage: getUsageTotals(),
        compactionCount: getCompactionCount(),
        // Client tool call detected (OpenResponses hosted tools)
        clientToolCall: clientToolCallDetected ?? undefined,
      };
    } finally {
      // Always tear down the session (and release the lock) before we leave this attempt.
      //
      // BUGFIX: Wait for the agent to be truly idle before flushing pending tool results.
      // pi-agent-core's auto-retry resolves waitForRetry() on assistant message receipt,
      // *before* tool execution completes in the retried agent loop. Without this wait,
      // flushPendingToolResults() fires while tools are still executing, inserting
      // synthetic "missing tool result" errors and causing silent agent failures.
      // See: https://github.com/openclaw/openclaw/issues/8643
      removeToolResultContextGuard?.();
      await flushPendingToolResultsAfterIdle({
        agent: session?.agent,
        sessionManager,
        clearPendingOnTimeout: true,
      });
      session?.dispose();
      releaseWsSession(params.sessionId);
      await sessionLock.release();
    }
  } finally {
    restoreSkillEnv?.();
    process.chdir(prevCwd);
  }
}
