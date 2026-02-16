'use client';

import Link from 'next/link';
import { Github, Database } from 'lucide-react';

export function Header() {
  return (
    <header className="sticky top-0 z-50 bg-white border-b border-gray-200">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-2">
              <Database className="w-5 h-5 text-red-600" />
              <span className="text-xl font-black tracking-tight">
                ENTITY DATABASE
              </span>
            </Link>
            <span className="hidden sm:inline text-xs text-gray-400 border-l pl-3 ml-1">
              by{' '}
              <a
                href="https://corp-o-rate.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-red-600 hover:underline"
              >
                corp-o-rate
              </a>
            </span>
          </div>

          {/* Navigation */}
          <nav className="flex items-center gap-4">
            <Link
              href="/docs"
              className="hidden sm:inline text-sm text-gray-600 hover:text-black transition-colors font-medium"
            >
              Docs
            </Link>
            <a
              href="#about"
              className="hidden sm:inline text-sm text-gray-600 hover:text-black transition-colors"
            >
              About
            </a>
            <a
              href="https://pypi.org/project/corp-entity-db/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-gray-600 hover:text-black transition-colors hidden sm:inline"
              title="View on PyPI"
            >
              PyPI
            </a>
            <a
              href="https://github.com/corp-o-rate/corp-entity-db"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 hover:bg-gray-100 rounded transition-colors"
              title="View on GitHub"
            >
              <Github className="w-5 h-5" />
            </a>
          </nav>
        </div>
      </div>
    </header>
  );
}