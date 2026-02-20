'use client';

import { useState } from 'react';
import { CodeBlock } from './code-block';

interface Tab {
  label: string;
  language?: string;
  code: string;
}

interface CodeTabsProps {
  tabs: Tab[];
}

export function CodeTabs({ tabs }: CodeTabsProps) {
  const [activeIndex, setActiveIndex] = useState(0);

  if (!tabs || tabs.length === 0) return null;

  const active = tabs[activeIndex];

  return (
    <div className="my-4">
      {/* Tab bar */}
      <div className="flex gap-0 bg-[#161616] rounded-t-lg border-b border-gray-800">
        {tabs.map((tab, i) => (
          <button
            key={tab.label}
            onClick={() => setActiveIndex(i)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              i === activeIndex
                ? 'text-white bg-[#1e1e1e] border-b-2 border-indigo-500'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {/* Code block without its own top rounding */}
      <div className="[&_.code-block-wrapper]:!my-0 [&_.code-block-wrapper>div:first-child]:!hidden [&_.code-block-wrapper_.code-block]:!rounded-t-none">
        <CodeBlock language={active.language || 'bash'}>{active.code}</CodeBlock>
      </div>
    </div>
  );
}
