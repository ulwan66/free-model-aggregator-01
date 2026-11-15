// The trial's brief and the env's instance document.
//
// The brief (at $MAKEFUN_PROMPT_FILE) carries three JSON documents buried in
// prose, found here by their `schema` member: the provider directory, the
// quality table, and the workload manifest. GET $MAKEFUN_ENV_URL/instance
// answers the same directory and manifest live; its manifest wins when the
// fetch works, the brief's otherwise. Everything here is a pure function of
// its inputs: no clock, no randomness.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider as the directory lists it. `path` is the catalog's base_url. */
export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** The provider's own path, such as /p/groq: the catalog reports it, dialling joins it to the env. */
  path: string;
  keyEnv: string;
  key: string;
}

/** One row of the quality table: a model's public rating and capabilities. */
export interface RatingRow {
  model: string;
  provider: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one request needs from the model that serves it. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface WorkloadEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: WorkloadEntry[];
  error: string | null;
}

function textOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function wireOf(v: unknown): Wire | null {
  return v === 'openai' || v === 'anthropic' || v === 'gemini' ? v : null;
}

function authOf(v: unknown): AuthScheme {
  if (v === 'bearer' || v === 'x-api-key' || v === 'x-goog-api-key' || v === 'none') return v;
  return 'bearer';
}

/** The path a directory entry names: `path` first, else the pathname of an absolute base_url. */
function pathOf(v: Json): string {
  const direct = textOf(v.path);
  if (direct !== '') return direct;
  for (const key of ['base_path', 'base_url', 'base']) {
    const raw = textOf(v[key]).trim();
    if (raw === '') continue;
    if (/^https?:\/\//i.test(raw)) {
      try {
        return new URL(raw).pathname.replace(/\/+$/, '') || '/';
      } catch {
        return raw;
      }
    }
    return raw;
  }
  return '';
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = textOf(v.id);
  const wire = wireOf(v.wire);
  const path = pathOf(v);
  if (id === '' || wire === null || path === '') return null;
  return {
    id,
    name: textOf(v.name) !== '' ? textOf(v.name) : id,
    wire,
    auth: authOf(v.auth),
    path,
    keyEnv: textOf(v.key_env),
    key: textOf(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = textOf(v.model);
  const provider = textOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    ratingMilli: numOrNull(v.rating_milli) ?? 0,
    contextLength: numOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadOf(v: unknown): WorkloadEntry | null {
  if (!isRecord(v)) return null;
  const id = textOf(v.id);
  if (id === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  const min = numOrNull(req.min_context);
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(req.tools === true ? { tools: true } : {}),
      ...(req.vision === true ? { vision: true } : {}),
      ...(req.json === true ? { json: true } : {}),
      ...(min !== null ? { minContext: min } : {}),
    },
  };
}

/** The index of the `}` closing the object that opens at `start`, or -1. Strings are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` carrying a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = objectEnd(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The brief at `path`: directory, quality table, manifest. A missing brief yields empty lists. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (dirDoc === undefined || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings =
    ratingsDoc !== undefined && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const manifest =
    workloadDoc !== undefined && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const w = workloadOf(v);
          return w === null ? [] : [w];
        })
      : [];
  return { providers, ratings, manifest, error: null };
}

/** GET /instance: the live directory and manifest. Null on any failure; the brief covers for it. */
export async function readInstance(
  envUrl: string,
  timeoutMs = 3000,
): Promise<{ providers: ProviderEntry[]; manifest: WorkloadEntry[] } | null> {
  if (envUrl === '') return null;
  try {
    const res = await fetch(`${envUrl}/instance`, { signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return null;
    const body: unknown = await res.json();
    if (!isRecord(body)) return null;
    const providers = Array.isArray(body.providers)
      ? body.providers.flatMap((v) => {
          const p = providerOf(v);
          return p === null ? [] : [p];
        })
      : [];
    const workload = isRecord(body.workload) && Array.isArray(body.workload.manifest) ? body.workload.manifest : [];
    const manifest = workload.flatMap((v) => {
      const w = workloadOf(v);
      return w === null ? [] : [w];
    });
    if (providers.length === 0 && manifest.length === 0) return null;
    return { providers, manifest };
  } catch {
    return null;
  }
}

/** The quality-table row for one provider's model, when the table has one. */
export function ratingRowFor(ratings: RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}
