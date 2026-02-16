'use client';

import { useState, useCallback } from 'react';
import { SearchBar, EntityType } from '@/components/SearchBar';
import { ResultCard } from '@/components/ResultCard';
import { Database, AlertCircle } from 'lucide-react';

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
      {/* Header */}
      <header className="border-b border-gray-200 px-4 sm:px-6 lg:px-8 py-4">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          <Database className="w-6 h-6 text-red-600" />
          <div>
            <h1 className="font-black text-lg tracking-tight">ENTITY DATABASE</h1>
            <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold">corp-o-rate</p>
          </div>
        </div>
      </header>

      <main className="flex-1 px-4 sm:px-6 lg:px-8 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Hero */}
          <div className="text-center mb-8">
            <span className="text-xs font-bold uppercase tracking-[0.3em] text-red-600">entity search</span>
            <h2 className="text-3xl md:text-4xl font-black mt-3 tracking-tight">
              SEARCH THE DATABASE.
            </h2>
            <p className="mt-3 text-gray-600 max-w-xl mx-auto">
              9.7M+ organizations and 63M+ people with USearch HNSW indexes for sub-millisecond lookups.
            </p>
          </div>

          {/* Search */}
          <div className="border-2 border-black shadow-[4px_4px_0_0_#000] bg-white p-6 mb-8">
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
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 px-4 sm:px-6 lg:px-8 py-4">
        <div className="max-w-4xl mx-auto text-center text-xs text-gray-400">
          corp-entity-db -- entity database search
        </div>
      </footer>
    </div>
  );
}
