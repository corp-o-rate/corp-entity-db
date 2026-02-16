import { NextRequest, NextResponse } from 'next/server';

const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const LOCAL_SERVER_URL = process.env.LOCAL_SERVER_URL;

const POLL_INTERVAL = 2000;
const MAX_POLLS = 30;

interface SearchRequest {
  query: string;
  type: 'org' | 'person' | 'role' | 'location';
  limit?: number;
  hybrid?: boolean;
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
          return NextResponse.json(data);
        }

        console.warn(`Local server returned ${localResponse.status}`);
      } catch (localError) {
        console.warn('Local server unavailable:', localError);
      }
    }

    // Try RunPod
    if (RUNPOD_ENDPOINT_ID && RUNPOD_API_KEY) {
      try {
        console.log(`Submitting search job to RunPod: type=${type} query="${query}"`);

        const runpodResponse = await fetch(
          `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run`,
          {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${RUNPOD_API_KEY}`,
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              input: { query, type, limit, hybrid },
            }),
          }
        );

        if (!runpodResponse.ok) {
          const errorText = await runpodResponse.text();
          console.error(`RunPod API error: status=${runpodResponse.status}, body=${errorText}`);
          throw new Error(`RunPod API error: ${runpodResponse.status}`);
        }

        const job = await runpodResponse.json();
        console.log(`RunPod job submitted: ${job.id}, status: ${job.status}`);

        // If completed immediately
        if (job.status === 'COMPLETED' && job.output) {
          return NextResponse.json(job.output);
        }

        // Poll for result
        for (let i = 0; i < MAX_POLLS; i++) {
          await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));

          const statusResponse = await fetch(
            `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${job.id}`,
            {
              headers: {
                'Authorization': `Bearer ${RUNPOD_API_KEY}`,
              },
            }
          );

          if (!statusResponse.ok) {
            console.error(`RunPod status check failed: ${statusResponse.status}`);
            continue;
          }

          const status = await statusResponse.json();
          console.log(`RunPod job ${job.id} status: ${status.status}`);

          if (status.status === 'COMPLETED') {
            return NextResponse.json(status.output);
          }

          if (status.status === 'FAILED') {
            throw new Error(status.error || 'RunPod job failed');
          }
        }

        return NextResponse.json(
          { error: 'Search timed out' },
          { status: 504 }
        );
      } catch (runpodError) {
        console.warn('RunPod unavailable:', runpodError);
        if (!LOCAL_SERVER_URL) {
          return NextResponse.json(
            { error: runpodError instanceof Error ? runpodError.message : 'RunPod error' },
            { status: 502 }
          );
        }
      }
    }

    // No backend available
    return NextResponse.json(
      { error: 'No search backend configured. Set LOCAL_SERVER_URL or RUNPOD_ENDPOINT_ID + RUNPOD_API_KEY.' },
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
