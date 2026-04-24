'use client';

import { Search, Loader2 } from 'lucide-react';

export type EntityType = 'org' | 'person' | 'role' | 'location';

interface SearchBarProps {
  query: string;
  onQueryChange: (query: string) => void;
  entityType: EntityType;
  onEntityTypeChange: (type: EntityType) => void;
  hybrid: boolean;
  onHybridChange: (hybrid: boolean) => void;
  onSubmit: () => void;
  onExampleClick: (example: string) => void;
  isLoading: boolean;
}

const entityTypes: { value: EntityType; label: string }[] = [
  { value: 'org', label: 'Organizations' },
  { value: 'person', label: 'People' },
  { value: 'role', label: 'Roles' },
  { value: 'location', label: 'Locations' },
];

const examples: Record<EntityType, string[]> = {
  org: ['Apple', 'Goldman Sachs', 'BlackRock', 'United Nations', 'MIT'],
  person: ['Tim Cook', 'Elon Musk', 'Barack Obama', 'Greta Thunberg', 'Taylor Swift'],
  role: ['CEO', 'Chief Financial Officer', 'Board Member', 'President', 'General Counsel'],
  location: ['California', 'London', 'New York', 'Tokyo', 'Delaware'],
};

export function SearchBar({
  query,
  onQueryChange,
  entityType,
  onEntityTypeChange,
  hybrid,
  onHybridChange,
  onSubmit,
  onExampleClick,
  isLoading,
}: SearchBarProps) {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && query.trim()) {
      onSubmit();
    }
  };

  return (
    <div className="space-y-4">
      {/* Entity type tabs */}
      <div className="flex border-b-2 border-gray-200">
        {entityTypes.map((type) => (
          <button
            key={type.value}
            onClick={() => onEntityTypeChange(type.value)}
            className={`px-4 py-2.5 text-sm font-semibold border-b-2 -mb-[2px] transition-colors cursor-pointer ${
              entityType === type.value
                ? 'border-red-600 text-black'
                : 'border-transparent text-gray-500 hover:text-black'
            }`}
          >
            {type.label}
          </button>
        ))}
      </div>

      {/* Search input row */}
      <div className="flex gap-3">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Search ${entityTypes.find(t => t.value === entityType)?.label.toLowerCase()}...`}
            className="w-full pl-10 pr-4 py-3 border border-gray-200 text-base focus:outline-none focus:border-black transition-colors"
            disabled={isLoading}
          />
        </div>
        <button
          onClick={onSubmit}
          disabled={isLoading || !query.trim()}
          className="px-6 py-3 bg-red-600 text-white font-bold hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer inline-flex items-center gap-2"
        >
          {isLoading ? (
            <Loader2 className="w-5 h-5 spinner" />
          ) : (
            <Search className="w-5 h-5" />
          )}
          Search
        </button>
      </div>

      {/* Quick examples */}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-gray-500 font-medium mr-1">Try:</span>
        {examples[entityType].map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => onExampleClick(example)}
            disabled={isLoading}
            className="px-3 py-1 border border-gray-200 text-gray-700 hover:border-black hover:text-black disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
          >
            {example}
          </button>
        ))}
      </div>

      {/* Hybrid toggle */}
      <label className="inline-flex items-center gap-2 text-sm text-gray-600 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={hybrid}
          onChange={(e) => onHybridChange(e.target.checked)}
          className="w-4 h-4 accent-red-600"
        />
        Hybrid search (text + embeddings)
      </label>
    </div>
  );
}
