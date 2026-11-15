// The catalog probe and the catalog document.
//
// Each provider's own models route is asked, in the provider's own wire, with
// the provider's own auth. The catalog that `--list` prints and /register
// carries is built once from those answers plus the quality table, so the two
// are the same document byte for byte. The catalog reports each provider's
// own path as its base_url, never the path joined to the env: the env's origin
// differs between a live pass and its replay, and the catalog is part of the
// state export a replay must reproduce.

import {
  isRecord,
  ratingRowFor,
  type AuthScheme,
  type Json,
  type ProviderEntry,
  type RatingRow,
  type Wire,
} from './brief.ts';

/** One model as a provider's listing wrote it; null members are what it did not say. */
export interface ProbedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  models: ProbedModel[];
  error: string | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register. */
export interface Catalog {
  providers: CatalogProvider[];
}

/** One model GET /v1/models lists: free models only, deduplicated by id. */
export interface ListedModel {
  id: string;
  provider: string;
  displayName: string;
  contextLength: number | null;
}

/** A path joined to the env's URL for dialling. */
export function dialBase(envUrl: string, path: string): string {
  const prefix = envUrl.replace(/\/+$/, '');
  const p = path.trim();
  if (p === '') return prefix;
  return `${prefix}${p.startsWith('/') ? '' : '/'}${p.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in; `none` sends no key at all. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function str(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? str(v.name).replace(/^models\//, '') : str(v.id);
  if (id === '') return null;
  const display =
    str(v.display_name) !== '' ? str(v.display_name) : str(v.displayName) !== '' ? str(v.displayName) : id;
  return {
    id,
    displayName: display,
    free: boolOrNull(v.free),
    contextLength: numOrNull(v.context_length) ?? numOrNull(v.context_window) ?? numOrNull(v.inputTokenLimit),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
export async function probeProvider(
  envUrl: string,
  provider: ProviderEntry,
  timeoutMs = 5000,
): Promise<Probe> {
  const base = dialBase(envUrl, provider.path);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/** Probe every provider in directory order, one at a time: deterministic, and kind to the env. */
export async function probeAll(envUrl: string, providers: ProviderEntry[]): Promise<Probe[]> {
  const out: Probe[] = [];
  for (const provider of providers) out.push(await probeProvider(envUrl, provider));
  return out;
}

/** Whether a model is free: the quality table's row where it has one, the listing's word otherwise. */
export function isFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** A model's capabilities: the quality table's row first, the listing's fields second. */
export function capabilities(model: ProbedModel, row: RatingRow | null): {
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
} {
  return {
    contextLength: row?.contextLength ?? model.contextLength ?? null,
    tools: row !== null ? row.tools : (model.tools ?? null),
    vision: row !== null ? row.vision : (model.vision ?? null),
    json: row !== null ? row.json : (model.json ?? null),
  };
}

/** The catalog: every provider that answered, every model it listed, models sorted by id. */
export function buildCatalog(probes: readonly Probe[], ratings: RatingRow[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const sorted = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowFor(ratings, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: isFree(model, row),
        context_length: row?.contextLength ?? model.contextLength ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.path,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

/** The free models GET /v1/models lists: deduplicated by id, sorted by id. */
export function listedModels(probes: readonly Probe[], ratings: RatingRow[]): ListedModel[] {
  const byId = new Map<string, ListedModel>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      if (byId.has(model.id)) continue;
      const row = ratingRowFor(ratings, probe.provider.id, model.id);
      if (!isFree(model, row)) continue;
      byId.set(model.id, {
        id: model.id,
        provider: probe.provider.id,
        displayName: model.displayName,
        contextLength: row?.contextLength ?? model.contextLength ?? null,
      });
    }
  }
  return [...byId.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

export type { Json };
