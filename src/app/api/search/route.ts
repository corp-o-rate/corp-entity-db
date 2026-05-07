import { NextRequest, NextResponse } from 'next/server';

const CEREBRIUM_ENDPOINT_URL = process.env.CEREBRIUM_ENDPOINT_URL;
const CEREBRIUM_TOKEN = process.env.CEREBRIUM_TOKEN;
const LOCAL_SERVER_URL = process.env.LOCAL_SERVER_URL;

interface SearchRequest {
  query: string;
  type: 'org' | 'person' | 'role' | 'location';
  limit?: number;
  hybrid?: boolean;
}

// The RunPod handler returns `{record: {...}, score}` per result with
// corp-entity-db's internal field names (known_for_role, known_for_org_name,
// region, source_id). The frontend (ResultCard.tsx) expects flat results with
// role/organization/country/wikidata_id. Normalize here so the frontend
// doesn't need to care about the backend's shape.
function normalizeResults(payload: unknown): unknown {
  if (!payload || typeof payload !== 'object') return payload;
  const p = payload as Record<string, unknown>;
  const results = p.results;
  if (!Array.isArray(results)) return payload;
  p.results = results.map((r) => {
    if (!r || typeof r !== 'object') return r;
    const item = r as Record<string, unknown>;
    const rec = (item.record ?? {}) as Record<string, unknown>;
    const source = rec.source as string | undefined;
    const sourceId = rec.source_id as string | undefined;
    return {
      ...rec,
      score: item.score,
      role: rec.known_for_role ?? rec.role,
      organization: rec.known_for_org_name ?? rec.organization,
      country: rec.country ?? rec.region,
      wikidata_id: source === 'wikidata' ? sourceId : undefined,
      canonical_id: (rec.record as { canon_id?: number } | undefined)?.canon_id,
    };
  });
  return payload;
}

export async function POST(request: NextRequest) {
  try {
    const body: SearchRequest = await request.json();
    const { query, type, limit = 20, hybrid = false } = body;

    if (!query || typeof query !== 'string') {
      return NextResponse.json(
        { error: 'Missing or invalid query' },
        { status: 400 }
      );
    }

    if (!['org', 'person', 'role', 'location'].includes(type)) {
      return NextResponse.json(
        { error: 'Invalid type. Must be org, person, role, or location.' },
        { status: 400 }
      );
    }

    // Try local server first
    if (LOCAL_SERVER_URL) {
      try {
        console.log(`Searching local server: ${LOCAL_SERVER_URL} type=${type} query="${query}"`);
        const localResponse = await fetch(`${LOCAL_SERVER_URL}/search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, type, limit, hybrid }),
        });

        if (localResponse.ok) {
          const data = await localResponse.json();
          return NextResponse.json(normalizeResults(data));
        }

        console.warn(`Local server returned ${localResponse.status}`);
      } catch (localError) {
        console.warn('Local server unavailable:', localError);
      }
    }

    // Try Cerebrium. Sync POST — handler runs to completion server-side.
    // Cerebrium wraps the response as { run_id, result, run_time_ms }; the
    // handler's actual payload is in `result`.
    if (CEREBRIUM_ENDPOINT_URL && CEREBRIUM_TOKEN) {
      try {
        console.log(`Calling Cerebrium: type=${type} query="${query}"`);
        const cerebriumResponse = await fetch(CEREBRIUM_ENDPOINT_URL, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${CEREBRIUM_TOKEN}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ query, type, limit, hybrid }),
        });

        if (!cerebriumResponse.ok) {
          const errorText = await cerebriumResponse.text();
          console.error(`Cerebrium error: status=${cerebriumResponse.status}, body=${errorText}`);
          throw new Error(`Cerebrium error: ${cerebriumResponse.status}`);
        }

        const envelope = await cerebriumResponse.json();
        console.log(`Cerebrium run ${envelope.run_id} completed in ${envelope.run_time_ms}ms`);
        const payload = envelope.result ?? envelope;
        return NextResponse.json(normalizeResults(payload));
      } catch (cerebriumError) {
        console.warn('Cerebrium unavailable:', cerebriumError);
        if (!LOCAL_SERVER_URL) {
          return NextResponse.json(
            { error: cerebriumError instanceof Error ? cerebriumError.message : 'Cerebrium error' },
            { status: 502 }
          );
        }
      }
    }

    // No backend available
    return NextResponse.json(
      { error: 'No search backend configured. Set LOCAL_SERVER_URL or CEREBRIUM_ENDPOINT_URL + CEREBRIUM_TOKEN.' },
      { status: 503 }
    );
  } catch (error) {
    console.error('Search API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
