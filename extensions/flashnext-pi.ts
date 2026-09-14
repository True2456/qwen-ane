// Registers the Flash-Next MIL ANE server (port 2457) as a Pi provider.
// This is not the native rindi binary on :2456.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const BASE_URL = "http://127.0.0.1:2457/v1";
const API_KEY = "flashnext";
const LEAN_SYSTEM_PROMPT = `You are a concise coding assistant running locally on Apple Silicon.
Answer the user's request directly. Inspect or modify files only when asked, and explain the result briefly.
Use the available tools when they are required to complete the request.`;

export default function (api: ExtensionAPI) {
  api.registerProvider("flashnext", {
    baseUrl: BASE_URL,
    apiKey: API_KEY,
    api: "openai-completions",
    authHeader: true,
    models: [
      {
        id: "Qwen3.8-Flash-Next",
        name: "Qwen3.8-Flash-Next (ANE MIL · :2457)",
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
        // The checkpoint template accepts only low / medium / xhigh. `null`
        // hides a Pi level; omitting max hides it. Off stays in the UI and
        // sets enable_thinking false.
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
    if (ctx.model?.provider !== "flashnext" || process.env.FLASHNEXT_FULL_SYSTEM_PROMPT) {
      return {};
    }
    return { systemPrompt: LEAN_SYSTEM_PROMPT };
  });
}
