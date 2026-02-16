'use client';

import { Building2, User, Briefcase, MapPin, Globe, Calendar, Hash } from 'lucide-react';
import type { EntityType } from './SearchBar';

interface SearchResult {
  name: string;
  score: number;
  type?: string;
  entity_type?: string;
  source?: string;
  country?: string;
  aliases?: string[];
  // Person fields
  role?: string;
  organization?: string;
  birth_date?: string;
  death_date?: string;
  person_type?: string;
  // Role fields
  role_name?: string;
  // Location fields
  location_type?: string;
  // Common
  id?: string | number;
  canonical_id?: string | number;
  wikidata_id?: string;
}

interface ResultCardProps {
  result: SearchResult;
  entityType: EntityType;
  rank: number;
}

const typeColors: Record<string, string> = {
  business: 'bg-blue-50 text-blue-700',
  fund: 'bg-indigo-50 text-indigo-700',
  branch: 'bg-slate-50 text-slate-700',
  nonprofit: 'bg-emerald-50 text-emerald-700',
  ngo: 'bg-teal-50 text-teal-700',
  foundation: 'bg-green-50 text-green-700',
  government: 'bg-red-50 text-red-700',
  international_org: 'bg-purple-50 text-purple-700',
  educational: 'bg-amber-50 text-amber-700',
  research: 'bg-cyan-50 text-cyan-700',
  healthcare: 'bg-rose-50 text-rose-700',
  media: 'bg-pink-50 text-pink-700',
  sports: 'bg-orange-50 text-orange-700',
  political_party: 'bg-red-50 text-red-600',
  trade_union: 'bg-yellow-50 text-yellow-700',
  // Person types
  executive: 'bg-violet-50 text-violet-700',
  politician: 'bg-red-50 text-red-700',
  athlete: 'bg-cyan-50 text-cyan-700',
  artist: 'bg-pink-50 text-pink-700',
  academic: 'bg-amber-50 text-amber-700',
  scientist: 'bg-teal-50 text-teal-700',
  journalist: 'bg-slate-50 text-slate-700',
  entrepreneur: 'bg-blue-50 text-blue-700',
  activist: 'bg-green-50 text-green-700',
  military: 'bg-gray-100 text-gray-700',
  legal: 'bg-indigo-50 text-indigo-700',
  professional: 'bg-stone-50 text-stone-700',
  media_person: 'bg-fuchsia-50 text-fuchsia-700',
};

function TypeBadge({ type }: { type: string }) {
  const colors = typeColors[type] || 'bg-gray-100 text-gray-600';
  return (
    <span className={`inline-flex items-center px-2 py-0.5 text-xs font-semibold uppercase tracking-wide ${colors}`}>
      {type.replace(/_/g, ' ')}
    </span>
  );
}

function SourceBadge({ source }: { source: string }) {
  return (
    <span className="inline-flex items-center px-2 py-0.5 text-xs font-medium bg-gray-100 text-gray-500 uppercase tracking-wide">
      {source}
    </span>
  );
}

function ScoreBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = pct >= 80 ? 'text-green-700' : pct >= 50 ? 'text-amber-600' : 'text-gray-500';
  return (
    <span className={`font-mono text-sm font-bold ${color}`}>
      {pct}%
    </span>
  );
}

function EntityIcon({ entityType }: { entityType: EntityType }) {
  switch (entityType) {
    case 'org': return <Building2 className="w-5 h-5 text-blue-500" />;
    case 'person': return <User className="w-5 h-5 text-violet-500" />;
    case 'role': return <Briefcase className="w-5 h-5 text-amber-500" />;
    case 'location': return <MapPin className="w-5 h-5 text-emerald-500" />;
  }
}

export function ResultCard({ result, entityType, rank }: ResultCardProps) {
  const displayType = result.entity_type || result.person_type || result.type || result.location_type;

  return (
    <div className="border border-gray-200 p-4 hover:border-black hover:shadow-sm transition-all">
      <div className="flex items-start gap-3">
        {/* Rank */}
        <span className="font-mono text-xs text-gray-400 mt-1 w-5 text-right shrink-0">
          {rank}
        </span>

        {/* Icon */}
        <div className="mt-0.5 shrink-0">
          <EntityIcon entityType={entityType} />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="font-bold text-base truncate">{result.name}</h3>
            <ScoreBadge score={result.score} />
          </div>

          {/* Badges row */}
          <div className="flex items-center gap-2 mt-1.5 flex-wrap">
            {displayType && <TypeBadge type={displayType} />}
            {result.source && <SourceBadge source={result.source} />}
          </div>

          {/* Person details */}
          {entityType === 'person' && (result.role || result.organization) && (
            <div className="flex items-center gap-1.5 mt-2 text-sm text-gray-600">
              <Briefcase className="w-3.5 h-3.5 shrink-0" />
              <span>
                {result.role}
                {result.role && result.organization && ' @ '}
                {result.organization && (
                  <span className="font-medium text-black">{result.organization}</span>
                )}
              </span>
            </div>
          )}

          {/* Dates for people */}
          {entityType === 'person' && (result.birth_date || result.death_date) && (
            <div className="flex items-center gap-1.5 mt-1 text-sm text-gray-500">
              <Calendar className="w-3.5 h-3.5 shrink-0" />
              <span className="font-mono text-xs">
                {result.birth_date || '?'}
                {result.death_date ? ` -- ${result.death_date}` : ''}
              </span>
            </div>
          )}

          {/* Org details */}
          {entityType === 'org' && result.country && (
            <div className="flex items-center gap-1.5 mt-2 text-sm text-gray-600">
              <Globe className="w-3.5 h-3.5 shrink-0" />
              <span>{result.country}</span>
            </div>
          )}

          {/* Aliases */}
          {result.aliases && result.aliases.length > 0 && (
            <div className="mt-2 text-xs text-gray-500">
              Also known as: {result.aliases.slice(0, 3).join(', ')}
              {result.aliases.length > 3 && ` +${result.aliases.length - 3} more`}
            </div>
          )}

          {/* IDs row */}
          {(result.wikidata_id || result.canonical_id) && (
            <div className="flex items-center gap-3 mt-2 text-xs text-gray-400">
              {result.wikidata_id && (
                <span className="inline-flex items-center gap-1 font-mono">
                  <Hash className="w-3 h-3" />{result.wikidata_id}
                </span>
              )}
              {result.canonical_id && (
                <span className="font-mono">id:{String(result.canonical_id)}</span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
