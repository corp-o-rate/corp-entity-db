'use client';

import { ExternalLink, Heart, Github } from 'lucide-react';

export function Footer() {
  return (
    <footer className="dark-section py-16 px-4 sm:px-6 lg:px-8 mt-16">
      {/* Gradient accent */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `
            radial-gradient(ellipse at 20% 50%, rgba(239, 68, 68, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 50%, rgba(249, 115, 22, 0.06) 0%, transparent 50%)
          `,
        }}
      />

      <div className="max-w-6xl mx-auto relative z-10">
        <div className="text-center">
          <span className="section-label">POWERED BY</span>
          <h2 className="text-3xl md:text-4xl font-black mt-4">corp-o-rate.com</h2>
          <p className="text-slate-400 mt-4 max-w-xl mx-auto">
            AI-powered corporate intelligence. Search 9.7M+ organizations and 63M+ people
            with embedding-based semantic search and USearch HNSW indexes.
          </p>

          <div className="mt-8 flex flex-wrap justify-center gap-4">
            <a
              href="https://corp-o-rate.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 bg-white text-black font-bold transition-all hover:-translate-y-0.5"
              style={{ boxShadow: '4px 4px 0 0 rgba(239, 68, 68, 0.6)' }}
            >
              Explore corp-o-rate.com
              <ExternalLink className="w-4 h-4" />
            </a>
            <a
              href="https://github.com/corp-o-rate/corp-entity-db"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 border border-white/20 text-white font-semibold hover:bg-white/5 transition-all"
            >
              <Github className="w-4 h-4" />
              View on GitHub
            </a>
          </div>
        </div>

        <div className="mt-12 pt-8 border-t border-white/10 flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-slate-500">
          <div className="flex items-center gap-1">
            Made with <Heart className="w-4 h-4 text-red-500" /> by the corp-o-rate team
          </div>
          <div className="flex items-center gap-4">
            <a
              href="https://pypi.org/project/corp-entity-db/"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-white transition-colors"
            >
              PyPI
            </a>
            <a
              href="https://huggingface.co/datasets/Corp-o-Rate-Community/entity-references"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-white transition-colors"
            >
              HuggingFace
            </a>
            <a
              href="https://corp-o-rate.com/privacy"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-white transition-colors"
            >
              Privacy
            </a>
            <a
              href="https://corp-o-rate.com/terms"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-white transition-colors"
            >
              Terms
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
