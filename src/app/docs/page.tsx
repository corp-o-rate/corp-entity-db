'use client';

import Link from 'next/link';
import { ArrowLeft, Book, Code2, Database, Rocket, Server, Terminal, Globe } from 'lucide-react';
import { TableOfContents, TocItem } from '@/components/docs/table-of-contents';

// Import documentation sections
import GettingStarted from './sections/getting-started.mdx';
import DatabaseSchema from './sections/database-schema.mdx';
import EntityTypes from './sections/entity-types.mdx';
import DataSources from './sections/data-sources.mdx';
import Cli from './sections/cli.mdx';
import PythonApi from './sections/python-api.mdx';
import ServerApi from './sections/server-api.mdx';
import Examples from './sections/examples.mdx';
import Deployment from './sections/deployment.mdx';

// Table of contents structure
const tocItems: TocItem[] = [
  {
    id: 'getting-started',
    label: 'Getting Started',
    level: 2,
    children: [
      { id: 'installation', label: 'Installation', level: 3 },
      { id: 'quick-start', label: 'Quick Start', level: 3 },
      { id: 'using-with-statement-extractor', label: 'Using with statement-extractor', level: 3 },
      { id: 'requirements', label: 'Requirements', level: 3 },
    ],
  },
  {
    id: 'database-schema',
    label: 'Database Schema',
    level: 2,
    children: [
      { id: 'schema-overview', label: 'Schema Overview', level: 3 },
      { id: 'enum-tables', label: 'Enum Lookup Tables', level: 3 },
      { id: 'embedding-storage', label: 'Embedding Storage', level: 3 },
      { id: 'usearch-indexes', label: 'USearch HNSW Indexes', level: 3 },
      { id: 'database-variants', label: 'Database Variants', level: 3 },
      { id: 'schema-version', label: 'Schema Version', level: 3 },
    ],
  },
  {
    id: 'entity-types',
    label: 'Entity Types',
    level: 2,
    children: [
      { id: 'organization-types', label: 'Organization Types', level: 3 },
      { id: 'person-types', label: 'Person Types', level: 3 },
      { id: 'location-types', label: 'Location Types', level: 3 },
      { id: 'source-types', label: 'Source Types', level: 3 },
    ],
  },
  {
    id: 'data-sources',
    label: 'Data Sources',
    level: 2,
    children: [
      { id: 'gleif', label: 'GLEIF', level: 3 },
      { id: 'sec-edgar', label: 'SEC EDGAR', level: 3 },
      { id: 'sec-form4', label: 'SEC Form 4', level: 3 },
      { id: 'companies-house', label: 'Companies House', level: 3 },
      { id: 'companies-house-officers', label: 'CH Officers', level: 3 },
      { id: 'wikidata-sparql', label: 'Wikidata (SPARQL)', level: 3 },
      { id: 'wikidata-dump', label: 'Wikidata Dump', level: 3 },
      { id: 'import-summary', label: 'Import Summary', level: 3 },
    ],
  },
  {
    id: 'cli',
    label: 'Command Line Interface',
    level: 2,
    children: [
      { id: 'commands-overview', label: 'Commands Overview', level: 3 },
      { id: 'global-options', label: 'Global Options', level: 3 },
      { id: 'search-commands', label: 'Search Commands', level: 3 },
      { id: 'import-commands', label: 'Import Commands', level: 3 },
      { id: 'management-commands', label: 'Management Commands', level: 3 },
      { id: 'serve-command', label: 'Serve Command', level: 3 },
    ],
  },
  {
    id: 'python-api',
    label: 'Python API',
    level: 2,
    children: [
      { id: 'organization-database', label: 'OrganizationDatabase', level: 3 },
      { id: 'person-database', label: 'PersonDatabase', level: 3 },
      { id: 'roles-database', label: 'RolesDatabase', level: 3 },
      { id: 'locations-database', label: 'LocationsDatabase', level: 3 },
      { id: 'company-embedder', label: 'CompanyEmbedder', level: 3 },
      { id: 'hub-functions', label: 'Hub Functions', level: 3 },
      { id: 'organization-resolver', label: 'OrganizationResolver', level: 3 },
      { id: 'data-models', label: 'Data Models', level: 3 },
      { id: 'entity-db-client', label: 'EntityDBClient', level: 3 },
    ],
  },
  {
    id: 'server-api',
    label: 'Server API',
    level: 2,
    children: [
      { id: 'starting-server', label: 'Starting the Server', level: 3 },
      { id: 'endpoints', label: 'Endpoints', level: 3 },
      { id: 'python-client', label: 'Python Client', level: 3 },
      { id: 'runpod-deployment', label: 'RunPod Deployment', level: 3 },
    ],
  },
  {
    id: 'examples',
    label: 'Examples',
    level: 2,
    children: [
      { id: 'search-organizations-example', label: 'Search Organizations', level: 3 },
      { id: 'search-people-example', label: 'Search People', level: 3 },
      { id: 'hybrid-search-example', label: 'Hybrid Search', level: 3 },
      { id: 'building-database', label: 'Building a Database', level: 3 },
      { id: 'pipeline-integration', label: 'Pipeline Integration', level: 3 },
      { id: 'server-delegation-example', label: 'Server Delegation', level: 3 },
      { id: 'batch-import-example', label: 'Batch Import', level: 3 },
    ],
  },
  {
    id: 'deployment',
    label: 'Deployment',
    level: 2,
    children: [
      { id: 'local-usage', label: 'Local Usage', level: 3 },
      { id: 'server-mode', label: 'Server Mode', level: 3 },
      { id: 'runpod-serverless', label: 'RunPod Serverless', level: 3 },
      { id: 'docker-setup', label: 'Docker Setup', level: 3 },
    ],
  },
];

export default function DocsPage() {
  return (
    <>
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                <span className="text-sm font-medium">Back to Search Demo</span>
              </Link>
              <div className="h-6 w-px bg-gray-200" />
              <div className="flex items-center gap-2">
                <Book className="w-5 h-5 text-indigo-600" />
                <span className="font-bold">Documentation</span>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <a
                href="https://pypi.org/project/corp-entity-db"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-gray-600 hover:text-indigo-600 transition-colors"
              >
                PyPI
              </a>
              <a
                href="https://huggingface.co/Corp-o-Rate-Community/entity-references"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-gray-600 hover:text-indigo-600 transition-colors"
              >
                HuggingFace
              </a>
              <a
                href="https://github.com/corp-o-rate/corp-entity-db"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-gray-600 hover:text-indigo-600 transition-colors"
              >
                GitHub
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="lg:grid lg:grid-cols-[1fr_250px] lg:gap-8">
          {/* Documentation content */}
          <main className="prose prose-gray max-w-none">
            {/* Hero */}
            <div className="not-prose mb-12">
              <span className="text-indigo-600 text-xs font-bold tracking-widest uppercase">
                corp-entity-db v0.1.0
              </span>
              <h1 className="text-4xl font-black mt-2 mb-4">
                Entity Database Documentation
              </h1>
              <p className="text-xl text-gray-600 max-w-2xl">
                Search and resolve organizations, people, roles, and locations across 9.7M+ organizations
                and 63M+ people using embedding-based USearch HNSW indexes.
              </p>

              {/* Quick links */}
              <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-4 mt-8">
                <a
                  href="#getting-started"
                  className="flex items-center gap-3 p-4 border-2 border-gray-200 hover:border-indigo-500 transition-colors group"
                >
                  <Rocket className="w-5 h-5 text-gray-400 group-hover:text-indigo-500" />
                  <div>
                    <div className="font-semibold">Getting Started</div>
                    <div className="text-sm text-gray-500">Installation & quick start</div>
                  </div>
                </a>
                <a
                  href="#cli"
                  className="flex items-center gap-3 p-4 border-2 border-gray-200 hover:border-indigo-500 transition-colors group"
                >
                  <Terminal className="w-5 h-5 text-gray-400 group-hover:text-indigo-500" />
                  <div>
                    <div className="font-semibold">CLI</div>
                    <div className="text-sm text-gray-500">Search & import commands</div>
                  </div>
                </a>
                <a
                  href="#database-schema"
                  className="flex items-center gap-3 p-4 border-2 border-gray-200 hover:border-indigo-500 transition-colors group"
                >
                  <Database className="w-5 h-5 text-gray-400 group-hover:text-indigo-500" />
                  <div>
                    <div className="font-semibold">Database Schema</div>
                    <div className="text-sm text-gray-500">v3 normalized schema</div>
                  </div>
                </a>
                <a
                  href="#data-sources"
                  className="flex items-center gap-3 p-4 border-2 border-gray-200 hover:border-indigo-500 transition-colors group"
                >
                  <Globe className="w-5 h-5 text-gray-400 group-hover:text-indigo-500" />
                  <div>
                    <div className="font-semibold">Data Sources</div>
                    <div className="text-sm text-gray-500">GLEIF, SEC, Wikidata & more</div>
                  </div>
                </a>
                <a
                  href="#python-api"
                  className="flex items-center gap-3 p-4 border-2 border-gray-200 hover:border-indigo-500 transition-colors group"
                >
                  <Code2 className="w-5 h-5 text-gray-400 group-hover:text-indigo-500" />
                  <div>
                    <div className="font-semibold">Python API</div>
                    <div className="text-sm text-gray-500">Classes & models</div>
                  </div>
                </a>
              </div>
            </div>

            {/* Documentation sections */}
            <GettingStarted />
            <DatabaseSchema />
            <EntityTypes />
            <DataSources />
            <Cli />
            <PythonApi />
            <ServerApi />
            <Examples />
            <Deployment />
          </main>

          {/* Table of contents sidebar */}
          <aside className="hidden lg:block">
            <TableOfContents items={tocItems} />
          </aside>
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-gray-200 mt-16 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between text-sm text-gray-500">
            <div>
              Built by{' '}
              <a href="https://corp-o-rate.com" className="text-indigo-600 hover:underline">
                Corp-o-Rate
              </a>
            </div>
            <div>MIT License</div>
          </div>
        </div>
      </footer>
    </>
  );
}
