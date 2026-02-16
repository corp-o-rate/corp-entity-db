'use client';

import { useEffect, useState } from 'react';
import { Menu, X } from 'lucide-react';

export interface TocItem {
  id: string;
  label: string;
  level: 1 | 2 | 3;
  children?: TocItem[];
}

interface TableOfContentsProps {
  items: TocItem[];
}

export function TableOfContents({ items }: TableOfContentsProps) {
  const [activeId, setActiveId] = useState<string>('');
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    // Intersection Observer for scroll spy
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
          }
        });
      },
      {
        rootMargin: '-80px 0px -80% 0px',
        threshold: 0,
      }
    );

    // Observe all heading elements
    const headings = document.querySelectorAll('h2[id], h3[id]');
    headings.forEach((heading) => observer.observe(heading));

    return () => {
      headings.forEach((heading) => observer.unobserve(heading));
    };
  }, []);

  const handleClick = (id: string) => {
    setActiveId(id);
    setMobileOpen(false);

    // Smooth scroll to element
    const element = document.getElementById(id);
    if (element) {
      const top = element.getBoundingClientRect().top + window.scrollY - 96;
      window.scrollTo({ top, behavior: 'smooth' });
    }
  };

  const renderItem = (item: TocItem) => {
    const isActive = activeId === item.id;
    const indent = item.level === 2 ? 'pl-0' : item.level === 3 ? 'pl-4' : 'pl-0';

    return (
      <li key={item.id}>
        <button
          onClick={() => handleClick(item.id)}
          className={`
            w-full text-left py-1.5 px-3 text-sm transition-all duration-150
            ${indent}
            ${isActive
              ? 'text-indigo-600 font-semibold border-l-2 border-indigo-600 bg-indigo-50'
              : 'text-gray-600 hover:text-gray-900 border-l-2 border-transparent hover:border-gray-300'
            }
          `}
        >
          {item.label}
        </button>
        {item.children && item.children.length > 0 && (
          <ul className="mt-1">
            {item.children.map(renderItem)}
          </ul>
        )}
      </li>
    );
  };

  return (
    <>
      {/* Mobile toggle button */}
      <button
        onClick={() => setMobileOpen(!mobileOpen)}
        className="lg:hidden fixed bottom-4 right-4 z-50 p-3 bg-black text-white rounded-full shadow-lg hover:bg-gray-800 transition-colors"
        aria-label="Toggle table of contents"
      >
        {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
      </button>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 z-40"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <nav
        className={`
          fixed lg:sticky top-0 lg:top-24 right-0 lg:right-auto
          w-72 lg:w-64 h-screen lg:h-auto max-h-[calc(100vh-6rem)]
          bg-white lg:bg-transparent
          border-l lg:border-l-0 border-gray-200
          overflow-y-auto
          z-40 lg:z-0
          transition-transform duration-300 lg:transform-none
          ${mobileOpen ? 'translate-x-0' : 'translate-x-full lg:translate-x-0'}
        `}
      >
        <div className="p-4 lg:p-0">
          <h4 className="font-bold text-xs uppercase tracking-wider text-gray-500 mb-4">
            On this page
          </h4>
          <ul className="space-y-1">
            {items.map(renderItem)}
          </ul>
        </div>
      </nav>
    </>
  );
}
