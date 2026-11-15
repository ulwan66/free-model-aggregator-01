// Routing: the best free model that can serve each request, learned as the trial goes.
//
// Budgets are drawn per trial and never published, and every provider refuses
// some of its calls on its own schedule, so the router keeps no fixed plan.
// For each request it ranks every provider that lists a free model meeting the
// request's requirements by that model's rating, less a penalty for each
// refusal the provider has just given, and tries them in order until one
// answers. A refusal marks the provider suspect; two in a row mark it dead
// for the rest of the trial unless nothing else is left. A provider is tried
// at most once per request: its next call number may succeed, but that retry
// belongs to a later request, not to this one's failover chain.
//
// A JSON-mode request from an OpenAI client can only go to a wire that carries
// JSON mode: the Anthropic wire echoes it absent, which fails the request.
// (Anthropic clients never carry JSON mode, so their JSON-capable requests
// have no such constraint.)

import { authHeaders, capabilities, dialBase, isFree, type Probe, type ProbedModel } from './catalog.ts';
import { ratingRowFor, type ProviderEntry, type RatingRow, type Requirement } from './brief.ts';
import {
  answerFrom,
  toAnthropic,
  toGemini,
  toOpenAI,
  type CanonAnswer,
  type CanonRequest,
} from './translate.ts';

/** Rating points subtracted per consecutive refusal when ranking a provider. */
const REFUSAL_PENALTY_MILLI = 25;
/** Rating points subtracted once a provider has served its minimum budget: the trial may have drawn
 *  exactly that minimum, so providers still below theirs go first. Never a ban: a provider past its
 *  minimum is still tried when it is the best option left. */
const MIN_BUDGET_PENALTY_MILLI = 60;
/**
 * Each provider's minimum per-trial budget, from the harness's published
 * budget ranges (the same table draws every trial, practice and scored).
 * Unknown providers get no prior: they are ranked on refusals alone.
 */
const MIN_BUDGET: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
/** Consecutive refusals after which a provider is dead unless nothing else is left. */
const DEAD_AFTER_REFUSALS = 2;
/** How long one provider call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;

export interface ProviderRuntime {
  entry: ProviderEntry;
  models: ProbedModel[];
  ratings: RatingRow[];
  /** Calls made to this provider so far, successes and refusals alike. */
  calls: number;
  /** Refusals since its last success. */
  refusals: number;
  dead: boolean;
}

export interface Candidate {
  runtime: ProviderRuntime;
  model: ProbedModel;
  ratingMilli: number;
  /** The rating less the refusal and minimum-budget penalties: what the ranking sorts by. */
  effective: number;
}

export function runtimeOf(probe: Probe, ratings: RatingRow[]): ProviderRuntime {
  return { entry: probe.provider, models: probe.models, ratings, calls: 0, refusals: 0, dead: false };
}

function meets(requires: Requirement, caps: ReturnType<typeof capabilities>): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.minContext !== undefined && (caps.contextLength === null || caps.contextLength < requires.minContext)) {
    return false;
  }
  return true;
}

/** The best free model on one provider that meets the requirements, or null. */
function bestModel(runtime: ProviderRuntime, requires: Requirement): { model: ProbedModel; ratingMilli: number } | null {
  let best: { model: ProbedModel; ratingMilli: number } | null = null;
  for (const model of runtime.models) {
    const row = ratingRowFor(runtime.ratings, runtime.entry.id, model.id);
    if (!isFree(model, row)) continue;
    if (!meets(requires, capabilities(model, row))) continue;
    const rating = row?.ratingMilli ?? 0;
    if (
      best === null ||
      rating > best.ratingMilli ||
      (rating === best.ratingMilli && model.id < best.model.id)
    ) {
      best = { model, ratingMilli: rating };
    }
  }
  return best;
}

/**
 * The ranked candidates for one request. `jsonWire` is true when the request
 * is a JSON-mode request from an OpenAI client: anthropic-wire providers are
 * then excluded, since their wire cannot echo JSON mode. Dead providers are
 * excluded unless that would leave no candidate at all.
 */
export function rankCandidates(
  runtimes: ProviderRuntime[],
  requires: Requirement,
  jsonWire: boolean,
): Candidate[] {
  const all: Candidate[] = [];
  for (const runtime of runtimes) {
    if (jsonWire && runtime.entry.wire === 'anthropic') continue;
    const best = bestModel(runtime, requires);
    if (best === null) continue;
    const min = MIN_BUDGET[runtime.entry.id];
    const overMinimum = min !== undefined && runtime.calls >= min;
    all.push({
      runtime,
      model: best.model,
      ratingMilli: best.ratingMilli,
      effective:
        best.ratingMilli -
        REFUSAL_PENALTY_MILLI * runtime.refusals -
        (overMinimum ? MIN_BUDGET_PENALTY_MILLI : 0),
    });
  }
  const live = all.filter((c) => !c.runtime.dead);
  const pool = live.length > 0 ? live : all;
  pool.sort(
    (a, b) =>
      b.effective - a.effective ||
      a.runtime.calls - b.runtime.calls ||
      (a.runtime.entry.id < b.runtime.entry.id ? -1 : a.runtime.entry.id > b.runtime.entry.id ? 1 : 0),
  );
  return pool;
}

export interface UpstreamResult {
  answer: CanonAnswer;
  providerId: string;
  modelId: string;
}

/** One inference call to one provider, in the provider's own wire. Null on any refusal or failure. */
async function callProvider(
  envUrl: string,
  runtime: ProviderRuntime,
  modelId: string,
  canon: CanonRequest,
): Promise<CanonAnswer | null> {
  const entry = runtime.entry;
  const base = dialBase(envUrl, entry.path);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(entry.auth, entry.key),
  };
  let url: string;
  let body: unknown;
  if (entry.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = toOpenAI(canon, modelId);
  } else if (entry.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = canon.anthropicVersion ?? '2023-06-01';
    body = toAnthropic(canon, modelId);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(modelId)}:generateContent`;
    body = toGemini(canon);
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
  if (!res.ok) {
    try {
      await res.arrayBuffer();
    } catch {
      // The body is drained on a best-effort basis only.
    }
    return null;
  }
  let parsed: unknown;
  try {
    parsed = await res.json();
  } catch {
    return null;
  }
  return answerFrom(entry.wire, parsed);
}

export interface ServeOutcome {
  result: UpstreamResult | null;
  /** The providers tried, in order, with whether each refused. */
  attempts: { providerId: string; modelId: string; refused: boolean }[];
}

/**
 * Serve one request: try the ranked candidates in order until one answers.
 * Provider state is updated as refusals and successes land, so the next
 * request's ranking reflects everything seen so far.
 */
export async function serve(
  envUrl: string,
  runtimes: ProviderRuntime[],
  requires: Requirement,
  canon: CanonRequest,
  jsonWire: boolean,
): Promise<ServeOutcome> {
  const attempts: ServeOutcome['attempts'] = [];
  const candidates = rankCandidates(runtimes, requires, jsonWire);
  for (const candidate of candidates) {
    const runtime = candidate.runtime;
    runtime.calls += 1;
    const answer = await callProvider(envUrl, runtime, candidate.model.id, canon);
    if (answer === null) {
      runtime.refusals += 1;
      if (runtime.refusals >= DEAD_AFTER_REFUSALS) runtime.dead = true;
      attempts.push({ providerId: runtime.entry.id, modelId: candidate.model.id, refused: true });
      continue;
    }
    runtime.refusals = 0;
    attempts.push({ providerId: runtime.entry.id, modelId: candidate.model.id, refused: false });
    return { result: { answer, providerId: runtime.entry.id, modelId: candidate.model.id }, attempts };
  }
  return { result: null, attempts };
}
