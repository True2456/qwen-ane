// Registers the Qwen3.8-27B pure-ANE server (port 1240) as a Pi provider.
// Independent of Flash-Next on :2457 and of native rindi on :2456.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const PORT = process.env.PURE27_PORT || "1240";
const BASE_URL = process.env.PURE27_BASE_URL || `http://127.0.0.1:${PORT}/v1`;
const API_KEY = "pure27";
const LEAN_SYSTEM_PROMPT = `You are a concise coding assistant running locally on Apple Silicon.
Answer the user's request directly. Inspect or modify files only when asked, and explain the result briefly.
Use the available tools when they are required to complete the request.`;

export default function (api: ExtensionAPI) {
  api.registerProvider("pure27", {
    baseUrl: BASE_URL,
    apiKey: API_KEY,
    api: "openai-completions",
    authHeader: true,
    models: [
      {
        id: "Qwen3.8-27B",
        name: "Qwen3.8-27B (pure ANE · :1240)",
        reasoning: true,
        input: ["text"] as "text"[],
        contextWindow: 131072,
        maxTokens: 2048,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        compat: {
          supportsStore: false,
          supportsDeveloperRole: false,
          supportsReasoningEffort: true,
          maxTokensField: "max_tokens",
          requiresReasoningContentOnAssistantMessages: true,
          thinkingFormat: "chat-template",
          chatTemplateKwargs: {
            enable_thinking: { $var: "thinking.enabled" },
            reasoning_effort: { $var: "thinking.effort", omitWhenOff: true },
            preserve_thinking: true,
          },
          supportsFinishReason: true,
        },
        thinkingLevelMap: {
          minimal: null,
          low: "low",
          medium: "medium",
          high: null,
          xhigh: "xhigh",
        },
      },
    ],
  });

  api.on("before_agent_start", async (_event: any, ctx) => {
    if (ctx.model?.provider !== "pure27" || process.env.PURE27_FULL_SYSTEM_PROMPT) {
      return {};
    }
    return { systemPrompt: LEAN_SYSTEM_PROMPT };
  });
}
