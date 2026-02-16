'use client';

import { Mail, Heart, Zap, Target, Users, Building2, Database, Globe, Search } from 'lucide-react';

export function HowItWorks() {
  return (
    <section className="py-16 px-4 sm:px-6 lg:px-8 border-t">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-10">
          <span className="section-label">TECHNICAL DETAILS</span>
          <h2 className="text-2xl md:text-3xl font-black mt-4">How It Works</h2>
        </div>

        {/* Architecture Overview */}
        <div className="mb-12">
          <h3 className="text-xl font-bold mb-6">Architecture</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-gray-50 p-4 rounded-lg border">
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Search className="w-4 h-4 text-red-600" />
                Embedding-Based Search
              </h4>
              <p className="text-sm text-gray-600">
                Uses google/embeddinggemma-300m (300M params) to generate dense vector embeddings.
                USearch HNSW indexes enable sub-millisecond approximate nearest neighbor lookups.
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg border">
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Database className="w-4 h-4 text-red-600" />
                Hybrid Search
              </h4>
              <p className="text-sm text-gray-600">
                Combines text filtering with embedding similarity for higher precision.
                SQLite with 256MB mmap, 500MB page cache, and WAL journal mode.
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg border">
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Globe className="w-4 h-4 text-red-600" />
                Multi-Source Data
              </h4>
              <p className="text-sm text-gray-600">
                ~9.9M organizations from GLEIF, SEC Edgar, Companies House, and Wikidata.
                ~66.9M people from Wikidata and Companies House officers.
                Canonicalization links equivalent records across sources.
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg border">
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Zap className="w-4 h-4 text-red-600" />
                Compact Storage
              </h4>
              <p className="text-sm text-gray-600">
                The lite database variant ships without embeddings &mdash; just the USearch HNSW
                indexes for fast ANN search. No need to download or store raw embedding vectors.
              </p>
            </div>
          </div>
        </div>

        {/* Data Sources Table */}
        <div className="mb-12">
          <h3 className="text-xl font-bold mb-6">Data Sources</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="bg-gray-100">
                  <th className="border border-gray-300 px-4 py-2 text-left">Source</th>
                  <th className="border border-gray-300 px-4 py-2 text-left">Description</th>
                  <th className="border border-gray-300 px-4 py-2 text-left">Scale</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border border-gray-300 px-4 py-2 font-bold">Companies House</td>
                  <td className="border border-gray-300 px-4 py-2">UK registered companies + officers</td>
                  <td className="border border-gray-300 px-4 py-2">~5.5M orgs, ~27.5M people</td>
                </tr>
                <tr className="bg-gray-50">
                  <td className="border border-gray-300 px-4 py-2 font-bold">Wikidata</td>
                  <td className="border border-gray-300 px-4 py-2">Organizations &amp; notable people</td>
                  <td className="border border-gray-300 px-4 py-2">~1.7M orgs, ~39.4M people</td>
                </tr>
                <tr>
                  <td className="border border-gray-300 px-4 py-2 font-bold">GLEIF</td>
                  <td className="border border-gray-300 px-4 py-2">Legal Entity Identifier records</td>
                  <td className="border border-gray-300 px-4 py-2">~2.6M orgs</td>
                </tr>
                <tr className="bg-gray-50">
                  <td className="border border-gray-300 px-4 py-2 font-bold">SEC Edgar</td>
                  <td className="border border-gray-300 px-4 py-2">US public company filers &amp; officers</td>
                  <td className="border border-gray-300 px-4 py-2">~73K orgs</td>
                </tr>
                <tr className="font-semibold bg-gray-100">
                  <td className="border border-gray-300 px-4 py-2">Total</td>
                  <td className="border border-gray-300 px-4 py-2">Organizations, people, roles &amp; locations</td>
                  <td className="border border-gray-300 px-4 py-2">~9.9M orgs, ~66.9M people</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Feedback CTA */}
        <div className="bg-gradient-to-r from-red-50 to-orange-50 border-2 border-red-200 rounded-lg p-8 text-center">
          <h3 className="text-xl font-bold mb-3 flex items-center justify-center gap-2">
            <Heart className="w-5 h-5 text-red-500" />
            We Need Your Feedback
          </h3>
          <p className="text-gray-700 mb-4 max-w-xl mx-auto">
            The entity database is actively being expanded. If you find missing organizations,
            incorrect data, or have suggestions for new data sources, we&apos;d love to hear from you.
          </p>
          <a
            href="mailto:neil@corp-o-rate.com"
            className="inline-flex items-center gap-2 px-6 py-3 bg-black text-white font-bold hover:bg-gray-800 transition-colors"
          >
            <Mail className="w-5 h-5" />
            neil@corp-o-rate.com
          </a>
        </div>
      </div>
    </section>
  );
}

export function AboutCorpORate() {
  return (
    <section id="about" className="py-16 px-4 sm:px-6 lg:px-8 bg-gray-900 text-white">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-10">
          <span className="text-red-400 text-xs font-bold tracking-widest uppercase">Who We Are</span>
          <h2 className="text-2xl md:text-3xl font-black mt-4">About Corp-o-Rate</h2>
        </div>

        {/* Mission */}
        <div className="mb-10">
          <div className="flex items-start gap-4 mb-6">
            <div className="bg-red-600 p-3 rounded-lg shrink-0">
              <Building2 className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-bold mb-2">The Glassdoor of ESG</h3>
              <p className="text-gray-300 text-lg">
                Real corporate intelligence from real people. Track what companies <em>actually do</em>, not what they claim.
              </p>
            </div>
          </div>

          <p className="text-gray-400 mb-4">
            Corp-o-Rate is building a community-powered corporate accountability platform. We believe that glossy
            sustainability reports and PR-polished ESG claims don&apos;t tell the full story. Our mission is to surface
            the truth about corporate behavior through crowdsourced intelligence, AI-powered analysis, and
            transparent data.
          </p>

          <p className="text-gray-400">
            The entity database is a core component of the Corp-o-Rate platform &mdash; providing fast, reliable
            entity resolution across 9.7M+ organizations and 63M+ people. Available as the{' '}
            <a
              href="https://pypi.org/project/corp-entity-db"
              target="_blank"
              rel="noopener noreferrer"
              className="text-red-400 hover:underline"
            >
              corp-entity-db
            </a>{' '}
            Python library on PyPI.
          </p>
        </div>

        {/* What we're building */}
        <div className="grid md:grid-cols-3 gap-6 mb-10">
          <div className="bg-gray-800 p-5 rounded-lg border border-gray-700">
            <Users className="w-8 h-8 text-red-400 mb-3" />
            <h4 className="font-bold mb-2">Community-Driven</h4>
            <p className="text-sm text-gray-400">
              Powered by employees, consumers, and researchers sharing real knowledge about corporate practices.
            </p>
          </div>
          <div className="bg-gray-800 p-5 rounded-lg border border-gray-700">
            <Zap className="w-8 h-8 text-red-400 mb-3" />
            <h4 className="font-bold mb-2">AI-Powered</h4>
            <p className="text-sm text-gray-400">
              Using NLP and knowledge graphs to structure, connect, and analyze corporate claims at scale.
            </p>
          </div>
          <div className="bg-gray-800 p-5 rounded-lg border border-gray-700">
            <Target className="w-8 h-8 text-red-400 mb-3" />
            <h4 className="font-bold mb-2">100% Independent</h4>
            <p className="text-sm text-gray-400">
              No corporate sponsors. No conflicts of interest. Just transparent corporate intelligence.
            </p>
          </div>
        </div>

        {/* Pre-funding notice */}
        <div className="bg-gradient-to-r from-red-900/50 to-orange-900/50 border border-red-700 rounded-lg p-8 text-center">
          <h3 className="text-xl font-bold mb-3">We&apos;re Pre-Funding &amp; Running on Fumes</h3>
          <p className="text-gray-300 mb-6 max-w-2xl mx-auto">
            Corp-o-Rate is currently bootstrapped and self-funded. We&apos;re building in public, shipping what we can,
            and working toward our mission one step at a time. If you believe in corporate accountability and
            transparent business intelligence, we&apos;d love your support.
          </p>

          <div className="flex flex-wrap justify-center gap-4 mb-6">
            <div className="bg-gray-800 px-4 py-2 rounded border border-gray-600">
              <span className="text-gray-400 text-sm">GPU Credits</span>
              <p className="font-bold text-white">Help us train better models</p>
            </div>
            <div className="bg-gray-800 px-4 py-2 rounded border border-gray-600">
              <span className="text-gray-400 text-sm">Angel Investment</span>
              <p className="font-bold text-white">Help us scale the platform</p>
            </div>
            <div className="bg-gray-800 px-4 py-2 rounded border border-gray-600">
              <span className="text-gray-400 text-sm">Partnerships</span>
              <p className="font-bold text-white">Data, research, or distribution</p>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a
              href="mailto:neil@corp-o-rate.com?subject=Corp-o-Rate%20Support"
              className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-red-600 text-white font-bold hover:bg-red-700 transition-colors"
            >
              <Mail className="w-5 h-5" />
              Get in Touch
            </a>
            <a
              href="https://corp-o-rate.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-white text-gray-900 font-bold hover:bg-gray-100 transition-colors"
            >
              Visit Corp-o-Rate
            </a>
          </div>
        </div>

        <p className="text-center text-gray-500 mt-8 text-sm">
          Shop smarter. Invest better. Know which companies match your values.
        </p>
      </div>
    </section>
  );
}
