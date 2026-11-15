// The wires: one canonical request, three upstream shapes, two client shapes.
//
// A client request is parsed once into a canonical form that keeps exactly
// what the providers echo back: the system text, each message's text parts in
// order, each image's bytes, the tool names in order, max_tokens, the stop
// sequences, and whether JSON mode was asked. Every upstream body is built
// from that form, so a request forwarded to any wire echoes exactly what the
// client sent. Upstream answers are always read as JSON (never streamed: the
// streamed and JSON answers carry the same text, usage and stop), buffered
// whole, and rendered downstream in the wire the client used, streamed or not
// as it asked.

import { isRecord, type Json, type Wire } from './brief.ts';

/** A stop reason's family: the mapping the brief fixes between wires. */
export type StopFamily = 'stop' | 'length' | 'tool_calls' | 'content_filter';

export interface CanonImage {
  /** A data image carries its bytes; a URL image carries its address. */
  kind: 'data' | 'url';
  mediaType: string;
  base64: string;
  url: string;
}

export interface CanonToolCall {
  id: string;
  name: string;
  argumentsText: string;
}

export interface CanonMessage {
  role: string;
  textParts: string[];
  images: CanonImage[];
  toolCallId: string;
  toolCalls: CanonToolCall[];
}

export interface CanonTool {
  name: string;
  description: string;
  parameters: Json | null;
}

export interface CanonRequest {
  systemText: string;
  messages: CanonMessage[];
  tools: CanonTool[];
  /** The client's tool choice, kept in the wire's own spelling for forwarding. */
  toolChoice: unknown;
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  /** The client's exact response_format object, forwarded to wires that carry it. */
  responseFormat: Json | null;
  temperature: number | null;
  topP: number | null;
  stream: boolean;
  /** The anthropic-version header the client sent, forwarded to anthropic-wire providers. */
  anthropicVersion: string | null;
}

export interface CanonAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: StopFamily;
}

function str(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The text of a string or of a block array's text blocks, concatenated with no separator. */
function answerTextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  let text = '';
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') text += block.text;
  }
  return text;
}

/** The text of a string or of a block array's text blocks, joined with newlines. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** The text parts of a message's content, in order. */
function textPartsOf(content: unknown): string[] {
  if (typeof content === 'string') return [content];
  if (!Array.isArray(content)) return [];
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts;
}

function imageOfDataUrl(url: string): CanonImage | null {
  const m = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(url);
  if (m === null) return null;
  if (m[2] === ';base64') return { kind: 'data', mediaType: m[1] !== '' ? m[1] : 'image/png', base64: m[3], url: '' };
  return { kind: 'url', mediaType: '', base64: '', url };
}

/** An OpenAI message's content blocks as canonical parts. */
function openAIMessage(raw: Json): CanonMessage {
  const role = str(raw.role) !== '' ? str(raw.role) : 'user';
  const msg: CanonMessage = { role, textParts: [], images: [], toolCallId: str(raw.tool_call_id), toolCalls: [] };
  const content = raw.content;
  msg.textParts = textPartsOf(content);
  if (Array.isArray(content)) {
    for (const block of content) {
      if (!isRecord(block) || block.type !== 'image_url' || !isRecord(block.image_url)) continue;
      const url = str(block.image_url.url);
      if (url === '') continue;
      msg.images.push(imageOfDataUrl(url) ?? { kind: 'url', mediaType: '', base64: '', url });
    }
  }
  if (Array.isArray(raw.tool_calls)) {
    for (const call of raw.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      msg.toolCalls.push({
        id: str(call.id),
        name: str(call.function.name),
        argumentsText: typeof call.function.arguments === 'string' ? call.function.arguments : '',
      });
    }
  }
  return msg;
}

/** An Anthropic message's content blocks as canonical parts. Tool results become tool messages. */
function anthropicMessages(source: unknown): CanonMessage[] {
  const out: CanonMessage[] = [];
  if (!Array.isArray(source)) return out;
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = str(raw.role) !== '' ? str(raw.role) : 'user';
    const msg: CanonMessage = { role, textParts: [], images: [], toolCallId: '', toolCalls: [] };
    const content = raw.content;
    msg.textParts = textPartsOf(content);
    if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'image' && isRecord(block.source)) {
          const src = block.source;
          if (src.type === 'base64' && typeof src.data === 'string') {
            msg.images.push({
              kind: 'data',
              mediaType: typeof src.media_type === 'string' ? src.media_type : 'image/png',
              base64: src.data,
              url: '',
            });
          } else if (src.type === 'url' && typeof src.url === 'string') {
            msg.images.push({ kind: 'url', mediaType: '', base64: '', url: src.url });
          }
        } else if (block.type === 'tool_use') {
          msg.toolCalls.push({
            id: str(block.id),
            name: str(block.name),
            argumentsText: isRecord(block.input) ? JSON.stringify(block.input) : '',
          });
        } else if (block.type === 'tool_result') {
          out.push({
            role: 'tool',
            textParts: [blocksText(block.content)],
            images: [],
            toolCallId: str(block.tool_use_id),
            toolCalls: [],
          });
        }
      }
    }
    out.push(msg);
  }
  return out;
}

/** An OpenAI Chat Completions body as a canonical request. */
export function parseOpenAIClient(body: Json, anthropicVersion: string | null): CanonRequest {
  const messages: CanonMessage[] = [];
  let systemText = '';
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    messages.push(openAIMessage(raw));
  }
  systemText = systemParts.join('\n');
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: str(tool.function.description),
        parameters: isRecord(tool.function.parameters) ? tool.function.parameters : null,
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format : null;
  const formatType = format !== null ? str(format.type) : '';
  return {
    systemText,
    messages,
    tools,
    toolChoice: body.tool_choice ?? null,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: formatType === 'json_object' || formatType === 'json_schema',
    responseFormat: format,
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stream: body.stream === true,
    anthropicVersion,
  };
}

/** An Anthropic Messages body as a canonical request. */
export function parseAnthropicClient(body: Json, anthropicVersion: string | null): CanonRequest {
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: str(tool.description),
        parameters: isRecord(tool.input_schema) ? tool.input_schema : null,
      });
    }
  }
  return {
    systemText: blocksText(body.system),
    messages: anthropicMessages(body.messages),
    tools,
    toolChoice: body.tool_choice ?? null,
    maxTokens: numOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    responseFormat: null,
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stream: body.stream === true,
    anthropicVersion,
  };
}

/** One canonical message's content on the OpenAI wire: a string, or text and image parts. */
function openAIContent(msg: CanonMessage): unknown {
  if (msg.images.length === 0 && msg.textParts.length <= 1) return msg.textParts[0] ?? '';
  const parts: Json[] = [];
  for (const text of msg.textParts) parts.push({ type: 'text', text });
  for (const image of msg.images) {
    const url = image.kind === 'data' ? `data:${image.mediaType};base64,${image.base64}` : image.url;
    parts.push({ type: 'image_url', image_url: { url } });
  }
  return parts;
}

/** One canonical message's content on the Anthropic wire: a string, or text and image blocks. */
function anthropicContent(msg: CanonMessage): unknown {
  if (msg.images.length === 0 && msg.textParts.length <= 1 && msg.toolCalls.length === 0) {
    return msg.textParts[0] ?? '';
  }
  const blocks: Json[] = [];
  for (const text of msg.textParts) blocks.push({ type: 'text', text });
  for (const image of msg.images) {
    if (image.kind === 'data') {
      blocks.push({ type: 'image', source: { type: 'base64', media_type: image.mediaType, data: image.base64 } });
    } else {
      blocks.push({ type: 'image', source: { type: 'url', url: image.url } });
    }
  }
  for (const call of msg.toolCalls) {
    let input: Json = {};
    if (call.argumentsText !== '') {
      try {
        const parsed: unknown = JSON.parse(call.argumentsText);
        if (isRecord(parsed)) input = parsed;
      } catch {
        // Not JSON: an empty input.
      }
    }
    blocks.push({ type: 'tool_use', id: call.id, name: call.name, input });
  }
  return blocks;
}

function openAITools(tools: CanonTool[]): Json[] {
  return tools.map((t) => ({
    type: 'function',
    function: {
      name: t.name,
      ...(t.description !== '' ? { description: t.description } : {}),
      ...(t.parameters !== null ? { parameters: t.parameters } : {}),
    },
  }));
}

function anthropicTools(tools: CanonTool[]): Json[] {
  return tools.map((t) => ({
    name: t.name,
    ...(t.description !== '' ? { description: t.description } : {}),
    input_schema: t.parameters ?? { type: 'object', properties: {} },
  }));
}

/** The client's tool choice rewritten for the OpenAI wire. */
function openAIToolChoice(choice: unknown): unknown {
  if (typeof choice === 'string') return choice;
  if (!isRecord(choice)) return undefined;
  if (choice.type === 'auto') return 'auto';
  if (choice.type === 'any') return 'required';
  if (choice.type === 'tool' && typeof choice.name === 'string') {
    return { type: 'function', function: { name: choice.name } };
  }
  return undefined;
}

/** The client's tool choice rewritten for the Anthropic wire. */
function anthropicToolChoice(choice: unknown): unknown {
  if (!isRecord(choice)) return undefined;
  if (choice.type === 'auto') return { type: 'auto' };
  if (choice.type === 'any') return { type: 'any' };
  if (choice.type === 'tool' && typeof choice.name === 'string') return { type: 'tool', name: choice.name };
  if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    return { type: 'tool', name: choice.function.name };
  }
  return undefined;
}

/** A canonical request as an OpenAI Chat Completions body naming `model`. Never streamed upstream. */
export function toOpenAI(canon: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  if (canon.systemText !== '') messages.push({ role: 'system', content: canon.systemText });
  for (const msg of canon.messages) {
    if (msg.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: msg.toolCallId, content: msg.textParts.join('\n') });
      continue;
    }
    const out: Json = { role: msg.role, content: openAIContent(msg) };
    if (msg.toolCalls.length > 0) {
      out.tool_calls = msg.toolCalls.map((c) => ({
        id: c.id,
        type: 'function',
        function: { name: c.name, arguments: c.argumentsText },
      }));
    }
    messages.push(out);
  }
  const body: Json = { model, messages };
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.temperature !== null) body.temperature = canon.temperature;
  if (canon.topP !== null) body.top_p = canon.topP;
  if (canon.tools.length > 0) body.tools = openAITools(canon.tools);
  const choice = openAIToolChoice(canon.toolChoice);
  if (choice !== undefined) body.tool_choice = choice;
  if (canon.jsonMode && canon.responseFormat !== null) body.response_format = canon.responseFormat;
  else if (canon.jsonMode) body.response_format = { type: 'json_object' };
  if (canon.stops.length > 0) body.stop = [...canon.stops];
  body.stream = false;
  return body;
}

/** A canonical request as an Anthropic Messages body naming `model`. Never streamed upstream. */
export function toAnthropic(canon: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  for (const msg of canon.messages) {
    if (msg.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: msg.toolCallId, content: msg.textParts.join('\n') }],
      });
      continue;
    }
    messages.push({ role: msg.role === 'assistant' ? 'assistant' : 'user', content: anthropicContent(msg) });
  }
  const body: Json = { model, messages };
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.systemText !== '') body.system = canon.systemText;
  if (canon.temperature !== null) body.temperature = canon.temperature;
  if (canon.topP !== null) body.top_p = canon.topP;
  if (canon.tools.length > 0) body.tools = anthropicTools(canon.tools);
  const choice = anthropicToolChoice(canon.toolChoice);
  if (choice !== undefined) body.tool_choice = choice;
  if (canon.stops.length > 0) body.stop_sequences = [...canon.stops];
  body.stream = false;
  return body;
}

/** A canonical request as a Gemini generateContent body. Tool messages are dropped: on this wire a
 *  non-model content overwrites the echoed user text, and a tool result is not user text. */
export function toGemini(canon: CanonRequest): Json {
  const contents: Json[] = [];
  for (const msg of canon.messages) {
    if (msg.role === 'tool') continue;
    const parts: Json[] = [];
    for (const text of msg.textParts) parts.push({ text });
    for (const image of msg.images) {
      if (image.kind === 'data') parts.push({ inlineData: { mimeType: image.mediaType, data: image.base64 } });
      else parts.push({ fileData: { fileUri: image.url } });
    }
    contents.push({ role: msg.role === 'assistant' ? 'model' : 'user', parts });
  }
  const body: Json = { contents };
  if (canon.systemText !== '') body.systemInstruction = { parts: [{ text: canon.systemText }] };
  if (canon.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: canon.tools.map((t) => ({
          name: t.name,
          ...(t.description !== '' ? { description: t.description } : {}),
          ...(t.parameters !== null ? { parameters: t.parameters } : {}),
        })),
      },
    ];
  }
  const config: Json = {};
  if (canon.maxTokens !== null) config.maxOutputTokens = canon.maxTokens;
  if (canon.stops.length > 0) config.stopSequences = [...canon.stops];
  if (canon.jsonMode) config.responseMimeType = 'application/json';
  if (canon.temperature !== null) config.temperature = canon.temperature;
  if (canon.topP !== null) config.topP = canon.topP;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

function stopOfOpenAI(reason: unknown): StopFamily {
  if (reason === 'length') return 'length';
  if (reason === 'tool_calls' || reason === 'function_call') return 'tool_calls';
  if (reason === 'content_filter') return 'content_filter';
  return 'stop';
}

function stopOfAnthropic(reason: unknown): StopFamily {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

function stopOfGemini(reason: unknown): StopFamily {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** An upstream JSON answer as a canonical answer, or null when it carries no usable text. */
export function answerFrom(wire: Wire, body: unknown): CanonAnswer | null {
  if (!isRecord(body)) return null;
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : {};
    const text = answerTextOf(message.content);
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text,
      inputTokens: numOrNull(usage.prompt_tokens) ?? 0,
      outputTokens: numOrNull(usage.completion_tokens) ?? 0,
      stop: stopOfOpenAI(choice.finish_reason),
    };
  }
  if (wire === 'anthropic') {
    const text = answerTextOf(body.content);
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text,
      inputTokens: numOrNull(usage.input_tokens) ?? 0,
      outputTokens: numOrNull(usage.output_tokens) ?? 0,
      stop: stopOfAnthropic(body.stop_reason),
    };
  }
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') text += part.text;
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: numOrNull(usage.promptTokenCount) ?? 0,
    outputTokens: numOrNull(usage.candidatesTokenCount) ?? 0,
    stop: stopOfGemini(candidate.finishReason),
  };
}

/** A fixed epoch second for every answer and listing: nothing here reads the clock. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** The width one streamed text delta carries. */
const DELTA_CHARS = 13;

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += DELTA_CHARS) pieces.push(text.slice(at, at + DELTA_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

function openAIStop(stop: StopFamily): string {
  if (stop === 'length') return 'length';
  if (stop === 'tool_calls') return 'tool_calls';
  if (stop === 'content_filter') return 'content_filter';
  return 'stop';
}

function anthropicStop(stop: StopFamily): string {
  if (stop === 'length') return 'max_tokens';
  if (stop === 'tool_calls') return 'tool_use';
  if (stop === 'content_filter') return 'refusal';
  return 'end_turn';
}

function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** A canonical answer as the client's JSON body. */
export function renderJson(client: 'openai' | 'anthropic', answer: CanonAnswer, model: string): Json {
  if (client === 'openai') {
    return {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion',
      created: FIXED_CREATED,
      model,
      choices: [
        {
          index: 0,
          message: { role: 'assistant', content: answer.text },
          finish_reason: openAIStop(answer.stop),
          logprobs: null,
        },
      ],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    };
  }
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A canonical answer as the client's SSE stream, in the wire's canonical event order. */
export function renderStream(client: 'openai' | 'anthropic', answer: CanonAnswer, model: string): string {
  if (client === 'openai') {
    const usage = {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    };
    const chunk = (delta: Json, finish: string | null): Json => ({
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
    const frames = [sseFrame(null, chunk({ role: 'assistant', content: '' }, null))];
    for (const piece of splitText(answer.text)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
    frames.push(sseFrame(null, chunk({}, openAIStop(answer.stop))));
    frames.push(
      sseFrame(null, {
        id: 'chatcmpl-aggregator',
        object: 'chat.completion.chunk',
        created: FIXED_CREATED,
        model,
        choices: [],
        usage,
      }),
    );
    frames.push('data: [DONE]\n\n');
    return frames.join('');
  }
  const frames = [
    sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseFrame('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(
      sseFrame('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(
    sseFrame('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseFrame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseFrame('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** An error in the client's wire. */
export function renderError(client: 'openai' | 'anthropic', type: string, message: string): Json {
  if (client === 'anthropic') return { type: 'error', error: { type, message } };
  return { error: { message, type, code: null } };
}

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(
  models: readonly { id: string; provider: string; contextLength: number | null }[],
): Json {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_CREATED,
      owned_by: m.provider,
      free: true,
      context_length: m.contextLength,
    })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(
  models: readonly { id: string; displayName: string; contextLength: number | null }[],
): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.displayName,
      created_at: FIXED_CREATED_AT,
      free: true,
      context_length: m.contextLength,
    })),
    has_more: false,
  };
}
