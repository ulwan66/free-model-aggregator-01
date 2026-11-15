// free-model-aggregator-01: one server answering both client wires from ten free providers.
//
// The trial runs this file as `node main.ts` with Node's standard library
// alone. It reads the brief, probes each provider's catalog route, builds one
// catalog, listens on 0.0.0.0 at a free port, registers with the env, and
// answers the client's requests until the env says the workload is done, then
// exits 0. `node main.ts --list` prints the same catalog and exits without
// starting the server.
//
// Every request is served by exactly one provider call: the best free model
// meeting the request's requirements, chosen from the quality table, with
// failover to the next candidate when a provider refuses. Requests are
// handled one at a time, in arrival order, so a provider never sees a second
// call while its first is in flight, and every choice is a function of the
// brief, the requests, and the providers' answers so far.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  isRecord,
  readBrief,
  readInstance,
  type Brief,
  type Json,
  type Requirement,
  type WorkloadEntry,
} from './lib/brief.ts';
import {
  buildCatalog,
  listedModels,
  probeAll,
  type Catalog,
  type ListedModel,
  type Probe,
} from './lib/catalog.ts';
import { runtimeOf, serve, type ProviderRuntime } from './lib/router.ts';
import {
  anthropicModelList,
  openAIModelList,
  parseAnthropicClient,
  parseOpenAIClient,
  renderError,
  renderJson,
  renderStream,
  type CanonRequest,
} from './lib/translate.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

let brief: Brief = { providers: [], ratings: [], manifest: [], error: null };
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
let runtimes: ProviderRuntime[] = [];
let models: ListedModel[] = [];
let manifest: WorkloadEntry[] = [];
/** Client inference requests handled so far: the manifest is in arrival order. */
let manifestIndex = 0;
/** Inference requests run one at a time, in arrival order. */
let queue: Promise<void> = Promise.resolve();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Read the brief, prefer the env's live manifest, probe the providers, build the catalog. */
async function load(): Promise<void> {
  brief = readBrief(PROMPT_FILE);
  if (brief.error !== null) log(brief.error);
  manifest = brief.manifest;
  const live = await readInstance(ENV_URL);
  if (live !== null && live.manifest.length > 0) manifest = live.manifest;
  probes = await probeAll(ENV_URL, brief.providers);
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
  runtimes = probes.filter((p) => p.ok).map((p) => runtimeOf(p, brief.ratings));
  models = listedModels(probes, brief.ratings);
}

/** The requirements of the request now arriving: the manifest's row, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    ...(Array.isArray(body.tools) && body.tools.length > 0 ? { tools: true } : {}),
    ...(vision ? { vision: true } : {}),
    ...(format === 'json_object' || format === 'json_schema' ? { json: true } : {}),
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendStream(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function sendError(res: ServerResponse, client: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  sendJson(res, status, renderError(client, type, message));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function listModels(req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(models));
  else sendJson(res, 200, openAIModelList(models));
}

/** POST /v1/chat/completions or POST /v1/messages: one request, one provider answer. */
async function infer(
  req: IncomingMessage,
  res: ServerResponse,
  client: 'openai' | 'anthropic',
): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, client, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const requires = requirementsFor(body);
  const version = typeof req.headers['anthropic-version'] === 'string' ? req.headers['anthropic-version'] : null;
  const canon: CanonRequest =
    client === 'openai' ? parseOpenAIClient(body, version) : parseAnthropicClient(body, version);
  // A JSON-mode request from an OpenAI client needs a wire that carries JSON mode.
  const jsonWire = client === 'openai' && canon.jsonMode;
  const { result, attempts } = await serve(ENV_URL, runtimes, requires, canon, jsonWire);
  if (result === null) {
    log(`no provider answered: ${attempts.map((a) => a.providerId).join(',') || 'no candidate'}`);
    sendError(res, client, 503, 'no_free_model', 'no free model answered this request');
    return;
  }
  log(
    `served by ${result.providerId}/${result.modelId} after ${attempts.length} attempt${attempts.length === 1 ? '' : 's'}`,
  );
  if (canon.stream) sendStream(res, renderStream(client, result.answer, result.modelId));
  else sendJson(res, 200, renderJson(client, result.answer, result.modelId));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    listModels(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await infer(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await infer(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog. */
async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') {
    log(`registered: the workload holds ${reply.requests} requests`);
  }
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    // Inference requests queue behind one another; the model list answers at once.
    if (req.method === 'POST') {
      queue = queue.then(() =>
        handle(req, res).catch((err: unknown) => {
          log((err as Error).stack ?? String(err));
          if (!res.headersSent) {
            sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
          }
        }),
      );
      return;
    }
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${models.length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
