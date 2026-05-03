#!/usr/bin/env bun
import { Clip, command, handler, invoke, z } from "@pinixai/core";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { extname, join } from "node:path";
import { homedir } from "node:os";

const DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions";
const DEFAULT_MODEL = "google/gemini-3-flash-preview";
const PINIX_SCHEME = "pinix://";

function dataDir(): string {
  const dir = process.env["PINIX_DATA_DIR"] ?? join(homedir(), ".pinix", "data", "perceive");
  mkdirSync(dir, { recursive: true });
  return dir;
}

function configPath(): string {
  return join(dataDir(), "config.json");
}

interface PerceiveConfig {
  api_key?: string;
  api_url?: string;
  default_model?: string;
}

function loadConfig(): PerceiveConfig {
  const p = configPath();
  if (!existsSync(p)) return {};
  return JSON.parse(readFileSync(p, "utf-8")) as PerceiveConfig;
}

function saveConfig(config: PerceiveConfig): void {
  writeFileSync(configPath(), JSON.stringify(config, null, 2) + "\n");
}

const MIME_MAP: Record<string, string> = {
  ".m4a": "audio/mp4",
  ".mp3": "audio/mpeg",
  ".wav": "audio/wav",
  ".ogg": "audio/ogg",
  ".flac": "audio/flac",
  ".aac": "audio/aac",
  ".webm": "audio/webm",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".svg": "image/svg+xml",
  ".bmp": "image/bmp",
  ".mp4": "video/mp4",
  ".mov": "video/quicktime",
  ".avi": "video/x-msvideo",
  ".mkv": "video/x-matroska",
  ".pdf": "application/pdf",
};

const DEFAULT_PROMPTS: Record<string, string> = {
  audio: "请完整转录这段音频的内容。保留原始语言，识别不同发言人。",
  image: "请详细描述这张图片的内容，包括文字、界面元素、数据或任何有价值的信息。",
  video: "请描述这段视频的内容，包括关键场景、对话和重要信息。",
};

function getApiKey(): string {
  const envKey = process.env["OPENROUTER_API_KEY"];
  if (envKey) return envKey;
  const config = loadConfig();
  if (config.api_key) return config.api_key;
  throw new Error("API key not configured. Run: perceive configure --api_key <key>");
}

function guessMime(filePath: string): string {
  const ext = extname(filePath).toLowerCase();
  const mime = MIME_MAP[ext];
  if (!mime) throw new Error(`Unsupported file type: ${ext}`);
  return mime;
}

function mediaCategory(mime: string): "audio" | "image" | "video" | "other" {
  if (mime.startsWith("audio/")) return "audio";
  if (mime.startsWith("image/")) return "image";
  if (mime.startsWith("video/")) return "video";
  return "other";
}

function defaultPrompt(mime: string): string {
  return DEFAULT_PROMPTS[mediaCategory(mime)] ?? DEFAULT_PROMPTS.image;
}

async function fetchFile(file: string): Promise<{ mime: string; base64: string }> {
  if (file.startsWith(PINIX_SCHEME)) {
    const filePath = file.slice(PINIX_SCHEME.length);
    const mime = guessMime(filePath);
    const result = (await invoke("fs", "cat", { path: filePath })) as {
      content: string;
      encoding: "utf-8" | "base64";
    };
    const base64 =
      result.encoding === "base64" ? result.content : Buffer.from(result.content).toString("base64");
    return { mime, base64 };
  }

  if (file.startsWith("http://") || file.startsWith("https://")) {
    const resp = await fetch(file);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
    const buf = await resp.arrayBuffer();
    const contentType = resp.headers.get("content-type")?.split(";")[0] ?? "";
    const mime = contentType || guessMime(file);
    return { mime, base64: Buffer.from(buf).toString("base64") };
  }

  throw new Error(`Unsupported URI scheme: ${file}. Use pinix:// or http(s)://`);
}

async function callGemini(
  model: string,
  mime: string,
  base64: string,
  prompt: string,
): Promise<string> {
  const apiKey = getApiKey();
  const config = loadConfig();
  const apiUrl = config.api_url ?? DEFAULT_API_URL;

  const payload = {
    model,
    messages: [
      {
        role: "user",
        content: [
          { type: "image_url", image_url: { url: `data:${mime};base64,${base64}` } },
          { type: "text", text: prompt },
        ],
      },
    ],
    max_tokens: 16384,
  };

  const resp = await fetch(apiUrl, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "https://yan5xu.ai",
    },
    body: JSON.stringify(payload),
  });

  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`Gemini API error (${resp.status}): ${body}`);
  }

  const result = (await resp.json()) as {
    choices: Array<{ message: { content: string } }>;
  };

  return result.choices[0].message.content;
}

class PerceiveClip extends Clip {
  name = "perceive";
  domain = [
    "Multimodal media perception — understand images, audio, and video with AI.",
    "Perceive is STATELESS: it has no memory between calls.",
    "Every call must be self-contained — include all relevant context in the prompt.",
    "For deeper understanding, use progressive multi-round analysis:",
    "  Round 1: rough scan → discover structure, terminology, patterns",
    "  Round 2: pass Round 1 findings as context → precise extraction",
    "  Round N: pass accumulated context + user corrections → final output",
  ].join("\n");
  patterns = [
    "analyze file + prompt → one-shot understanding",
    "analyze file + context (prior findings) + prompt → progressive deep analysis",
    "analyze pinix://recordings/meeting.m4a (context: 'Speaker A is the interviewer...') 'Extract key decisions' → precise extraction",
    "configure --api_key <key> → set API key",
  ];

  @command("Set or view configuration (api_key, api_url, default_model)")
  configure = handler(
    z.object({
      api_key: z.string().optional().describe("OpenRouter API key"),
      api_url: z.string().optional().describe("API endpoint URL"),
      default_model: z.string().optional().describe("Default model"),
    }),
    z.object({
      api_key: z.string().describe("API key (masked)"),
      api_url: z.string().describe("API endpoint URL"),
      default_model: z.string().describe("Default model"),
    }),
    async ({ api_key, api_url, default_model }) => {
      const config = loadConfig();
      if (api_key) config.api_key = api_key;
      if (api_url) config.api_url = api_url;
      if (default_model) config.default_model = default_model;
      if (api_key || api_url || default_model) saveConfig(config);
      const key = config.api_key ?? "";
      return {
        api_key: key ? `${key.slice(0, 6)}...${key.slice(-4)}` : "(not set)",
        api_url: config.api_url ?? DEFAULT_API_URL,
        default_model: config.default_model ?? DEFAULT_MODEL,
      };
    },
  );

  @command("Analyze a media file (image, audio, or video) with AI. Prompt should include full context — perceive is stateless and has no memory between calls. For progressive analysis, pass prior findings in context.")
  analyze = handler(
    z.object({
      file: z.string().describe("File URI: pinix://path or http(s)://url"),
      prompt: z.string().optional().describe("Analysis instruction (auto-detected by media type if omitted). Include all relevant context: terminology, prior findings, position info."),
      context: z.string().optional().describe("Accumulated context from prior rounds. Terminology, structure discovered in Round 1, user corrections, domain knowledge. This is prepended to the prompt to give the model full situational awareness."),
      model: z.string().default(DEFAULT_MODEL).describe("Model to use"),
    }),
    z.object({
      content: z.string().describe("AI analysis result"),
      model: z.string().describe("Model used"),
      file: z.string().describe("Original file reference"),
      mime: z.string().describe("Detected MIME type"),
    }),
    async ({ file, prompt, context, model }) => {
      const config = loadConfig();
      const actualModel = model === DEFAULT_MODEL && config.default_model ? config.default_model : model;
      const { mime, base64 } = await fetchFile(file);
      const basePrompt = prompt ?? defaultPrompt(mime);
      const fullPrompt = context ? `<context>\n${context}\n</context>\n\n${basePrompt}` : basePrompt;
      const content = await callGemini(actualModel, mime, base64, fullPrompt);
      return { content, model: actualModel, file, mime };
    },
  );
}

if (import.meta.main) {
  await new PerceiveClip().start();
}
