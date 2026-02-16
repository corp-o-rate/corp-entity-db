'use client';

import { useState, useEffect } from 'react';
import { Copy, Check, Terminal } from 'lucide-react';

interface CodeBlockProps {
  children: string;
  language?: string;
  filename?: string;
  showLineNumbers?: boolean;
}

export function CodeBlock({ children, language = 'text', filename, showLineNumbers = false }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const [highlightedHtml, setHighlightedHtml] = useState<string | null>(null);

  // Clean up the code (remove leading/trailing whitespace from the whole block)
  const code = typeof children === 'string' ? children.trim() : '';

  useEffect(() => {
    // Dynamically import and highlight code
    const highlight = async () => {
      try {
        const { highlightCode } = await import('@/lib/shiki');
        const html = await highlightCode(code, language as any);
        setHighlightedHtml(html);
      } catch (error) {
        console.error('Failed to highlight code:', error);
        // Fallback to plain text
        setHighlightedHtml(null);
      }
    };

    highlight();
  }, [code, language]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Get language display name
  const languageLabel = {
    python: 'Python',
    typescript: 'TypeScript',
    javascript: 'JavaScript',
    bash: 'Bash',
    xml: 'XML',
    json: 'JSON',
    sql: 'SQL',
    text: 'Text',
  }[language] || language;

  return (
    <div className="code-block-wrapper group relative my-4">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-[#161616] border-b border-gray-800 rounded-t-lg">
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <Terminal className="w-4 h-4" />
          {filename ? (
            <span className="font-mono">{filename}</span>
          ) : (
            <span>{languageLabel}</span>
          )}
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2 py-1 text-xs text-gray-400 hover:text-white transition-colors rounded hover:bg-gray-700"
          title="Copy to clipboard"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-green-500" />
              <span className="text-green-500">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code content */}
      <div className="code-block overflow-x-auto rounded-t-none">
        {highlightedHtml ? (
          <div
            className="shiki-wrapper [&_pre]:!bg-transparent [&_pre]:!p-0 [&_pre]:!m-0 [&_code]:!bg-transparent"
            dangerouslySetInnerHTML={{ __html: highlightedHtml }}
          />
        ) : (
          <pre className={showLineNumbers ? 'line-numbers' : ''}>
            <code>{code}</code>
          </pre>
        )}
      </div>
    </div>
  );
}

// Pre component for MDX - wraps code blocks automatically
export function Pre({ children, ...props }: React.ComponentProps<'pre'>) {
  // Extract language and code from children
  const codeElement = children as React.ReactElement<{ className?: string; children?: string }>;
  const className = codeElement?.props?.className || '';
  const language = className.replace(/language-/, '') || 'text';
  const code = codeElement?.props?.children || '';

  return <CodeBlock language={language}>{code}</CodeBlock>;
}
