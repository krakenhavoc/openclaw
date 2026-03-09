import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { resolveImplicitProviders } from "./models-config.providers.js";

const AZURE_BASE_URL = "https://my-resource.services.ai.azure.com";
const AZURE_API_KEY = "test-azure-key";

describe("Azure Foundry auto-discovery", () => {
  let originalVitest: string | undefined;
  let originalNodeEnv: string | undefined;
  let originalFetch: typeof globalThis.fetch;

  afterEach(() => {
    if (originalVitest !== undefined) {
      process.env.VITEST = originalVitest;
    } else {
      delete process.env.VITEST;
    }
    if (originalNodeEnv !== undefined) {
      process.env.NODE_ENV = originalNodeEnv;
    } else {
      delete process.env.NODE_ENV;
    }
    globalThis.fetch = originalFetch;
  });

  function setupDiscoveryEnv() {
    originalVitest = process.env.VITEST;
    originalNodeEnv = process.env.NODE_ENV;
    delete process.env.VITEST;
    delete process.env.NODE_ENV;
    originalFetch = globalThis.fetch;
  }

  function makeAzureDeploymentsResponse(
    deployments: Array<{ id: string; model: string; status: string }>,
  ) {
    return { data: deployments };
  }

  function mockFetchForAzure(deploymentsResponse: ReturnType<typeof makeAzureDeploymentsResponse>) {
    const fetchMock = vi.fn().mockImplementation(async (url: string | URL) => {
      if (String(url).includes("/openai/deployments")) {
        return {
          ok: true,
          json: async () => deploymentsResponse,
        };
      }
      // Reject any other fetch (e.g. Ollama probing)
      throw new Error(`connect ECONNREFUSED`);
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    return fetchMock;
  }

  it("discovers Azure deployments when models array is empty", async () => {
    setupDiscoveryEnv();
    const response = makeAzureDeploymentsResponse([
      { id: "gpt-4o-deploy", model: "gpt-4o", status: "succeeded" },
      { id: "gpt-35-turbo", model: "gpt-3.5-turbo", status: "succeeded" },
    ]);
    mockFetchForAzure(response);

    const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
    const providers = await resolveImplicitProviders({
      agentDir,
      explicitProviders: {
        "azure-foundry": {
          baseUrl: AZURE_BASE_URL,
          apiKey: AZURE_API_KEY,
          models: [],
        },
      },
    });

    expect(providers?.["azure-foundry"]).toBeDefined();
    expect(providers?.["azure-foundry"]?.models).toHaveLength(2);
    expect(providers?.["azure-foundry"]?.models?.[0]?.id).toBe("gpt-4o-deploy");
    expect(providers?.["azure-foundry"]?.models?.[0]?.name).toBe("gpt-4o (gpt-4o-deploy)");
    expect(providers?.["azure-foundry"]?.models?.[0]?.input).toEqual(["text", "image"]);
    expect(providers?.["azure-foundry"]?.models?.[1]?.id).toBe("gpt-35-turbo");
    expect(providers?.["azure-foundry"]?.api).toBe("openai-completions");
  });

  it("skips discovery when explicit models are provided", async () => {
    setupDiscoveryEnv();
    const fetchMock = mockFetchForAzure(
      makeAzureDeploymentsResponse([{ id: "gpt-4o-deploy", model: "gpt-4o", status: "succeeded" }]),
    );

    const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
    await resolveImplicitProviders({
      agentDir,
      explicitProviders: {
        "azure-foundry": {
          baseUrl: AZURE_BASE_URL,
          apiKey: AZURE_API_KEY,
          models: [
            {
              id: "my-manual-model",
              name: "Manual Model",
              reasoning: false,
              input: ["text"],
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
              contextWindow: 128_000,
              maxTokens: 4096,
            },
          ],
        },
      },
    });

    // Should not have called the deployments endpoint
    const azureCalls = fetchMock.mock.calls.filter((args: unknown[]) =>
      String(args[0]).includes("/openai/deployments"),
    );
    expect(azureCalls).toHaveLength(0);
  });

  it("filters out non-succeeded deployments", async () => {
    setupDiscoveryEnv();
    const response = makeAzureDeploymentsResponse([
      { id: "active-deploy", model: "gpt-4o", status: "succeeded" },
      { id: "failed-deploy", model: "gpt-4o", status: "failed" },
      { id: "creating-deploy", model: "gpt-4o", status: "creating" },
    ]);
    mockFetchForAzure(response);

    const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
    const providers = await resolveImplicitProviders({
      agentDir,
      explicitProviders: {
        "azure-foundry": {
          baseUrl: AZURE_BASE_URL,
          apiKey: AZURE_API_KEY,
          models: [],
        },
      },
    });

    expect(providers?.["azure-foundry"]?.models).toHaveLength(1);
    expect(providers?.["azure-foundry"]?.models?.[0]?.id).toBe("active-deploy");
  });

  it("handles API errors gracefully", async () => {
    setupDiscoveryEnv();
    globalThis.fetch = vi.fn().mockImplementation(async (url: string | URL) => {
      if (String(url).includes("/openai/deployments")) {
        return { ok: false, status: 401 };
      }
      throw new Error("connect ECONNREFUSED");
    }) as unknown as typeof fetch;

    const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
    const providers = await resolveImplicitProviders({
      agentDir,
      explicitProviders: {
        "azure-foundry": {
          baseUrl: AZURE_BASE_URL,
          apiKey: AZURE_API_KEY,
          models: [],
        },
      },
    });

    // Should not register provider when discovery fails
    expect(providers?.["azure-foundry"]).toBeUndefined();
  });

  it("uses api-key header for authentication", async () => {
    setupDiscoveryEnv();
    const fetchMock = mockFetchForAzure(
      makeAzureDeploymentsResponse([{ id: "my-deploy", model: "gpt-4o", status: "succeeded" }]),
    );

    const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
    await resolveImplicitProviders({
      agentDir,
      explicitProviders: {
        "azure-foundry": {
          baseUrl: AZURE_BASE_URL,
          apiKey: AZURE_API_KEY,
          models: [],
        },
      },
    });

    const azureCall = fetchMock.mock.calls.find((args: unknown[]) =>
      String(args[0]).includes("/openai/deployments"),
    );
    expect(azureCall).toBeDefined();
    const [, init] = azureCall as [string, RequestInit];
    const headers = init.headers as Record<string, string>;
    expect(headers["api-key"]).toBe(AZURE_API_KEY);
    expect(headers.Authorization).toBeUndefined();
  });

  it("infers reasoning from model name", async () => {
    setupDiscoveryEnv();
    const response = makeAzureDeploymentsResponse([
      { id: "o1-deploy", model: "o1-preview", status: "succeeded" },
      { id: "o3-deploy", model: "o3-mini", status: "succeeded" },
      { id: "gpt4-deploy", model: "gpt-4o", status: "succeeded" },
    ]);
    mockFetchForAzure(response);

    const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
    const providers = await resolveImplicitProviders({
      agentDir,
      explicitProviders: {
        "azure-foundry": {
          baseUrl: AZURE_BASE_URL,
          apiKey: AZURE_API_KEY,
          models: [],
        },
      },
    });

    const models = providers?.["azure-foundry"]?.models ?? [];
    expect(models).toHaveLength(3);
    expect(models[0]?.reasoning).toBe(true); // o1-preview
    expect(models[1]?.reasoning).toBe(true); // o3-mini
    expect(models[2]?.reasoning).toBe(false); // gpt-4o
  });

  it("also detects classic Azure OpenAI URLs", async () => {
    setupDiscoveryEnv();
    const response = makeAzureDeploymentsResponse([
      { id: "my-deploy", model: "gpt-4", status: "succeeded" },
    ]);
    mockFetchForAzure(response);

    const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
    const providers = await resolveImplicitProviders({
      agentDir,
      explicitProviders: {
        "my-azure": {
          baseUrl: "https://my-resource.openai.azure.com",
          apiKey: AZURE_API_KEY,
          models: [],
        },
      },
    });

    expect(providers?.["my-azure"]).toBeDefined();
    expect(providers?.["my-azure"]?.models).toHaveLength(1);
  });
});
