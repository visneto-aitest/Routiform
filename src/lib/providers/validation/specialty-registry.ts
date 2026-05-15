import { validateQoderCliPat } from "@routiform/open-sse/services/qoderCli.ts";
import type { JsonRecord } from "./constants";
import {
  validateAssemblyAIProvider,
  validateDeepgramProvider,
  validateElevenLabsProvider,
  validateInworldProvider,
  validateNanoBananaProvider,
} from "./specialty-audio-media";
import { validateBailianCodingPlanProvider } from "./specialty-bailian";
import {
  validateLongcatProvider,
  validateNvidiaProvider,
  validateXiaomiMimoTokenPlanProvider,
} from "./specialty-misc";
import { SEARCH_VALIDATOR_CONFIGS, validateSearchProvider } from "./search";
import { validateVertexProvider } from "./vertex";
import { safeOutboundFetch } from "@/lib/network/safeOutboundFetch";
import { buildBearerHeaders, toValidationErrorMessage } from "./http-utils";
import { randomUUID } from "crypto";

type SpecialtyValidatorResult = {
  valid: boolean;
  error: string | null;
  unsupported?: boolean;
  method?: string;
  warning?: string;
};

export const SPECIALTY_VALIDATORS: Record<
  string,
  (params: Record<string, unknown>) => Promise<SpecialtyValidatorResult>
> = {
  qoder: ({ apiKey, providerSpecificData }: Record<string, unknown>) =>
    validateQoderCliPat({
      apiKey: String(apiKey || ""),
      providerSpecificData: providerSpecificData as JsonRecord,
    }),
  deepgram: validateDeepgramProvider,
  assemblyai: validateAssemblyAIProvider,
  nanobanana: validateNanoBananaProvider,
  elevenlabs: validateElevenLabsProvider,
  inworld: validateInworldProvider,
  "bailian-coding-plan": validateBailianCodingPlanProvider,
  vertex: validateVertexProvider,
  nvidia: validateNvidiaProvider,
  longcat: validateLongcatProvider,
  "xiaomi-mimo-token-plan": validateXiaomiMimoTokenPlanProvider,
  commandcode: async ({ apiKey }: Record<string, unknown>) => {
    try {
      const res = await safeOutboundFetch(
        "https://api.commandcode.ai/alpha/generate",
        {
          method: "POST",
          headers: {
            ...buildBearerHeaders(String(apiKey || ""), {}),
            "x-session-id": randomUUID(),
            "x-command-code-version": "0.25.7",
            "x-cli-environment": "cli",
          },
          body: JSON.stringify({
            threadId: randomUUID(),
            memory: "",
            config: {
              workingDir: "/",
              date: new Date().toISOString().slice(0, 10),
              environment: "linux",
              structure: [],
              isGitRepo: false,
              currentBranch: "",
              mainBranch: "",
              gitStatus: "",
              recentCommits: [],
            },
            params: {
              model: "moonshotai/Kimi-K2.5",
              messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
              stream: true,
              max_tokens: 1,
              temperature: 0,
            },
          }),
        },
        { timeoutMs: 15_000 }
      );
      if (res.status === 401 || res.status === 403) {
        return { valid: false, error: "Invalid API key", unsupported: false };
      }
      if (res.status === 200 || res.status === 201 || res.status === 400 || res.status === 422) {
        return { valid: true, error: null };
      }
      return { valid: false, error: `Unexpected status ${res.status}`, unsupported: false };
    } catch (error: unknown) {
      return {
        valid: false,
        error: toValidationErrorMessage(error, "Validation failed"),
        unsupported: false,
      };
    }
  },
  ...Object.fromEntries(
    Object.entries(SEARCH_VALIDATOR_CONFIGS).map(([id, configFn]) => [
      id,
      ({ apiKey, providerSpecificData }: Record<string, unknown>) => {
        const { url, init } = configFn(String(apiKey || ""));
        return validateSearchProvider(url, init, providerSpecificData as JsonRecord);
      },
    ])
  ),
};
