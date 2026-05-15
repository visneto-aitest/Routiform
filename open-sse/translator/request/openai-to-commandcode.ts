/**
 * OpenAI → CommandCode request translator
 *
 * Upstream `/alpha/generate` schema:
 *  - params.system: STRING at top level (system messages NOT in messages[])
 *  - params.messages[*].role ∈ {"user","assistant","tool"}
 *  - params.messages[*].content: Array of content blocks (NEVER a string)
 *  - tool_use blocks (assistant): {type:"tool-call", toolCallId, toolName, input}
 *  - tool_result blocks (role=user): {type:"tool-result", toolCallId, toolName, output}
 *  - tools[*]: Anthropic plain {name, description, input_schema}
 */
import { randomUUID } from "crypto";

import { FORMATS } from "../formats.ts";
import { register } from "../registry.ts";

type JsonRecord = Record<string, unknown>;

type OpenAIMessage = {
  role?: string;
  content?: string | Array<{ type?: string; text?: string; image_url?: unknown }> | null;
  tool_calls?: Array<{
    id?: string;
    function?: { name?: string; arguments?: string | JsonRecord };
  }>;
  tool_call_id?: string;
  name?: string;
};

type OpenAITool = {
  type?: string;
  function?: { name?: string; description?: string; parameters?: JsonRecord };
  name?: string;
  description?: string;
  input_schema?: JsonRecord;
  parameters?: JsonRecord;
};

type ContentBlock =
  | { type: "text"; text: string }
  | { type: "tool-call"; toolCallId: string; toolName: string; input: JsonRecord }
  | {
      type: "tool-result";
      toolCallId: string;
      toolName: string;
      output: { type: "text"; value: string };
    };

type CommandCodeMessage = {
  role: "user" | "assistant" | "tool";
  content: ContentBlock[];
};

type CommandCodeTool = {
  name: string | undefined;
  description: string | undefined;
  input_schema: JsonRecord;
};

function flattenText(content: OpenAIMessage["content"]): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const p of content) {
      if (typeof p === "string") parts.push(p);
      else if (p && typeof p === "object" && typeof p.text === "string") parts.push(p.text);
    }
    return parts.join("\n");
  }
  return String(content);
}

function toContentBlocks(content: OpenAIMessage["content"]): ContentBlock[] {
  if (content == null) return [{ type: "text", text: "" }];
  if (typeof content === "string") return [{ type: "text", text: content }];
  if (Array.isArray(content)) {
    const blocks: ContentBlock[] = [];
    for (const part of content) {
      if (typeof part === "string") {
        blocks.push({ type: "text", text: part });
      } else if (part && typeof part === "object") {
        if (part.type === "text" && typeof part.text === "string") {
          blocks.push({ type: "text", text: part.text });
        } else if (part.type === "image_url" || part.type === "image") {
          blocks.push({ type: "text", text: "[image omitted]" });
        } else if (typeof part.text === "string") {
          blocks.push({ type: "text", text: part.text });
        }
      }
    }
    return blocks.length ? blocks : [{ type: "text", text: "" }];
  }
  return [{ type: "text", text: String(content) }];
}

function safeParseJson(s: string | JsonRecord | undefined): JsonRecord {
  if (s == null) return {};
  if (typeof s !== "string") return s as JsonRecord;
  try {
    return JSON.parse(s) as JsonRecord;
  } catch {
    return {};
  }
}

function convertMessages(messages: OpenAIMessage[]): {
  messages: CommandCodeMessage[];
  system: string;
} {
  const out: CommandCodeMessage[] = [];
  const systemTexts: string[] = [];

  for (const m of messages) {
    if (!m) continue;
    const role = m.role;

    if (role === "system") {
      const t = flattenText(m.content);
      if (t) systemTexts.push(t);
      continue;
    }

    if (role === "tool") {
      const value = typeof m.content === "string" ? m.content : flattenText(m.content);
      out.push({
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: m.tool_call_id || "",
            toolName: m.name || "",
            output: { type: "text", value },
          },
        ],
      });
      continue;
    }

    if (role === "assistant") {
      const blocks: ContentBlock[] = [];
      const text = flattenText(m.content);
      if (text) blocks.push({ type: "text", text });
      if (Array.isArray(m.tool_calls)) {
        for (const tc of m.tool_calls) {
          const fn = tc.function || {};
          blocks.push({
            type: "tool-call",
            toolCallId: tc.id || "",
            toolName: fn.name || "",
            input: safeParseJson(fn.arguments),
          });
        }
      }
      out.push({
        role: "assistant",
        content: blocks.length ? blocks : [{ type: "text", text: "" }],
      });
      continue;
    }

    out.push({ role: "user", content: toContentBlocks(m.content) });
  }

  return { messages: out, system: systemTexts.join("\n\n") };
}

function convertTools(tools: unknown): CommandCodeTool[] | undefined {
  if (!Array.isArray(tools) || tools.length === 0) return undefined;
  const result: CommandCodeTool[] = [];
  for (const t of tools as OpenAITool[]) {
    if (!t) continue;
    if (t.type === "function" && t.function) {
      result.push({
        name: t.function.name,
        description: t.function.description,
        input_schema: t.function.parameters || { type: "object" },
      });
    } else if (t.name && (t.input_schema || t.parameters)) {
      result.push({
        name: t.name,
        description: t.description,
        input_schema: t.input_schema || t.parameters || { type: "object" },
      });
    }
  }
  return result.length ? result : undefined;
}

export function openaiToCommandCode(
  model: string,
  body: Record<string, unknown>
): Record<string, unknown> {
  const rawMessages = Array.isArray(body.messages) ? (body.messages as OpenAIMessage[]) : [];
  const { messages, system } = convertMessages(rawMessages);

  const params: Record<string, unknown> = {
    model,
    messages,
    stream: true, // CommandCode requires stream:true; non-streaming is handled by SSE→JSON conversion downstream
    max_tokens: (body.max_tokens ?? body.max_output_tokens ?? 64000) as number,
    temperature: (body.temperature ?? 0.3) as number,
  };

  if (system) params.system = system;

  const tools = convertTools(body.tools);
  if (tools) params.tools = tools;
  if (body.top_p != null) params.top_p = body.top_p;

  const today = new Date().toISOString().slice(0, 10);

  return {
    threadId: randomUUID(),
    memory: "",
    config: {
      workingDir: process.cwd(),
      date: today,
      environment: process.platform,
      structure: [],
      isGitRepo: false,
      currentBranch: "",
      mainBranch: "",
      gitStatus: "",
      recentCommits: [],
    },
    params,
  };
}

register(FORMATS.OPENAI, FORMATS.COMMANDCODE, openaiToCommandCode);
