'use client';

import { useState, useCallback } from 'react';
import { Header } from '@/components/header';
import { Footer } from '@/components/footer';
import { SearchBar, EntityType } from '@/components/SearchBar';
import { ResultCard } from '@/components/ResultCard';
import { HowItWorks, AboutCorpORate } from '@/components/about-sections';
import { Database, AlertCircle, BookOpen, Terminal } from 'lucide-react';

interface SearchResult {
  name: string;
  score: number;
  type?: string;
  entity_type?: string;
  source?: string;
  country?: string;
  aliases?: string[];
  role?: string;
  organization?: string;
  birth_date?: string;
  death_date?: string;
  person_type?: string;
  role_name?: string;
  location_type?: string;
  id?: string | number;
  canonical_id?: string | number;
  wikidata_id?: string;
}

export default function Home() {
  const [query, setQuery] = useState('');
  const [entityType, setEntityType] = useState<EntityType>('org');
  const [hybrid, setHybrid] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [searchTime, setSearchTime] = useState<number | null>(null);

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setHasSearched(true);
    const start = performance.now();

    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          type: entityType,
          limit: 20,
          hybrid,
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || `Search failed (${response.status})`);
      }

      const data = await response.json();
      setResults(data.results || []);
      setSearchTime(Math.round(performance.now() - start));
    } catch (err) {
      console.error('Search error:', err);
      setError(err instanceof Error ? err.message : 'Search failed');
      setResults([]);
      setSearchTime(null);
    } finally {
      setIsLoading(false);
    }
  }, [query, entityType, hybrid]);

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1">
        {/* Hero Section */}
        <section className="py-12 px-4 sm:px-6 lg:px-8 border-b">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-8">
              <span className="section-label">corp-entity-db demo</span>
              <h1 className="text-4xl md:text-5xl font-black mt-4 tracking-tight">
                SEARCH THE DATABASE.
                <br />
                <span className="text-gray-400">RESOLVE ENTITIES.</span>
              </h1>
              <p className="mt-4 text-gray-600 max-w-2xl mx-auto">
                A Python library for searching and resolving organizations, people, roles, and locations.
                9.7M+ organizations and 63M+ people with embedding-based{' '}
                <a
                  href="https://github.com/unum-cloud/usearch"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-red-600 hover:underline font-medium"
                >
                  USearch HNSW
                </a>{' '}
                indexes for fast lookups. Data sourced from GLEIF, SEC Edgar,
                UK Companies House, and Wikidata.
              </p>
            </div>

            {/* Search Section */}
            <div className="brutal-card p-6 md:p-8">
              <SearchBar
                query={query}
                onQueryChange={setQuery}
                entityType={entityType}
                onEntityTypeChange={setEntityType}
                hybrid={hybrid}
                onHybridChange={setHybrid}
                onSubmit={handleSearch}
                isLoading={isLoading}
              />
            </div>
          </div>
        </section>

        {/* Results Section */}
        <section className="py-12 px-4 sm:px-6 lg:px-8">
          <div className="max-w-6xl mx-auto">
            {/* Error */}
            {error && (
              <div className="flex items-center gap-2 p-4 mb-6 border border-red-200 bg-red-50 text-red-700 text-sm">
                <AlertCircle className="w-4 h-4 shrink-0" />
                {error}
              </div>
            )}

            {/* Results header */}
            {hasSearched && !error && (
              <div className="flex items-center justify-between mb-4">
                <p className="text-sm text-gray-500">
                  {results.length} result{results.length !== 1 ? 's' : ''}
                  {searchTime !== null && (
                    <span className="font-mono ml-1">({searchTime}ms)</span>
                  )}
                </p>
              </div>
            )}

            {/* Results */}
            {results.length > 0 && (
              <div className="space-y-3">
                {results.map((result, i) => (
                  <ResultCard
                    key={`${result.name}-${result.id ?? i}`}
                    result={result}
                    entityType={entityType}
                    rank={i + 1}
                  />
                ))}
              </div>
            )}

            {/* Empty state */}
            {hasSearched && !isLoading && !error && results.length === 0 && (
              <div className="text-center py-16 text-gray-400">
                <Database className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p className="text-lg font-semibold">No results found</p>
                <p className="text-sm mt-1">Try a different query or entity type</p>
              </div>
            )}

            {/* Initial state */}
            {!hasSearched && (
              <div className="text-center py-16 text-gray-300">
                <Database className="w-16 h-16 mx-auto mb-4 opacity-20" />
                <p className="text-gray-400">Enter a query above to search</p>
              </div>
            )}
          </div>
        </section>

        {/* Quick Start Section */}
        <section className="py-12 px-4 sm:px-6 lg:px-8 bg-gray-50/50">
          <div className="max-w-6xl mx-auto">
            <div className="flex items-center gap-2 mb-6">
              <BookOpen className="w-5 h-5 text-red-600" />
              <h2 className="font-bold text-xl">Quick Start</h2>
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-gray-500" />
                  Install &amp; Search
                </h3>
                <pre className="code-block rounded-lg text-sm"><code>{`\
`}<span className="text-gray-500"># Install from PyPI</span>{`
pip install corp-entity-db

`}<span className="text-gray-500"># Download lite database + indexes</span>{`
corp-entity-db download

`}<span className="text-gray-500"># Search organizations</span>{`
corp-entity-db search "Microsoft"
corp-entity-db search "Microsoft" --hybrid

`}<span className="text-gray-500"># Search people</span>{`
corp-entity-db search-people "Tim Cook"`}</code></pre>
              </div>
              <div>
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-gray-500" />
                  Python API
                </h3>
                <pre className="code-block rounded-lg text-sm"><code><span className="text-green-400">from</span>{` corp_entity_db `}<span className="text-green-400">import</span>{` (
    OrganizationDatabase,
    get_database_path
)

db = OrganizationDatabase(
    get_database_path()
)
matches = db.search(
    `}<span className="text-yellow-300">&quot;Microsoft&quot;</span>{`, limit=`}<span className="text-cyan-300">10</span>{`
)
`}<span className="text-green-400">for</span>{` m `}<span className="text-green-400">in</span>{` matches:
    print(m.record.name, m.score)`}</code></pre>
              </div>
            </div>
          </div>
        </section>

        {/* How It Works Section */}
        <HowItWorks />

        {/* About Corp-o-Rate Section */}
        <AboutCorpORate />
      </main>

      <Footer />
    </div>
  );
}
