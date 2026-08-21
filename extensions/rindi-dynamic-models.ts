// Registers the Rindi Hybrid ANE+GPU engine (port 2456) as a provider in Pi.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const BASE_URL = "http://127.0.0.1:2456/v1";
const API_KEY = "rindi";

export default function (api: ExtensionAPI) {
  api.registerProvider("rindi", {
    baseUrl: BASE_URL,
    apiKey: API_KEY,
    api: "openai-completions",
    authHeader: true,
    async refreshModels() {
      let resp: Response;
      try {
        resp = await fetch(`${BASE_URL}/models`, {
          headers: { Authorization: `Bearer ${API_KEY}` },
          signal: AbortSignal.timeout(3000),
        });
      } catch {
        return [];
      }
      if (!resp.ok) return [];
      const data = (await resp.json()) as { data?: Array<{ id: string }> };
      const rawModels = data.data ?? [];

      return rawModels.map((m) => {
        const id = m.id;
        return {
          id,
          name: `${id} (Rindi Hybrid ANE+GPU)`,
          reasoning: true,
          input: ["text"] as ("text")[],
          contextWindow: 262144,
          maxTokens: 32768,
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
              reasoning_effort: { $var: "thinking.effort" },
              preserve_thinking: false,
            },
          },
          thinkingLevelMap: {
            off: "low",
            minimal: "low",
            low: "low",
            medium: "medium",
            high: "xhigh",
            xhigh: "xhigh",
            max: "xhigh",
          },
        };
      });
    },
  });
}
