'use client';

import { Pre, CodeBlock } from './code-block';
import { MermaidCodeBlock } from './mermaid';
import Link from 'next/link';
import { ExternalLink } from 'lucide-react';

// MDX component types
type MDXComponents = Record<string, React.ComponentType<any>>;

// Custom components for MDX content
export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    // Headings with anchor links
    h1: ({ children, ...props }) => (
      <h1 className="text-3xl font-black mt-12 mb-6 first:mt-0" {...props}>
        {children}
      </h1>
    ),
    h2: ({ children, id, ...props }) => (
      <h2
        id={id}
        className="text-2xl font-bold mt-10 mb-4 pb-2 border-b border-gray-200 scroll-mt-24"
        {...props}
      >
        <a href={`#${id}`} className="hover:text-indigo-600 transition-colors">
          {children}
        </a>
      </h2>
    ),
    h3: ({ children, id, ...props }) => (
      <h3
        id={id}
        className="text-xl font-semibold mt-8 mb-3 scroll-mt-24"
        {...props}
      >
        <a href={`#${id}`} className="hover:text-indigo-600 transition-colors">
          {children}
        </a>
      </h3>
    ),
    h4: ({ children, id, ...props }) => (
      <h4
        id={id}
        className="text-lg font-semibold mt-6 mb-2 scroll-mt-24"
        {...props}
      >
        {children}
      </h4>
    ),

    // Paragraphs
    p: ({ children, ...props }) => (
      <p className="my-4 leading-7 text-gray-700" {...props}>
        {children}
      </p>
    ),

    // Links
    a: ({ href, children, ...props }) => {
      const isExternal = href?.startsWith('http');
      if (isExternal) {
        return (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-indigo-600 hover:underline inline-flex items-center gap-1"
            {...props}
          >
            {children}
            <ExternalLink className="w-3 h-3" />
          </a>
        );
      }
      return (
        <Link href={href || '#'} className="text-indigo-600 hover:underline" {...props}>
          {children}
        </Link>
      );
    },

    // Code blocks
    pre: ({ children, ...props }) => {
      // Check if this is a mermaid code block
      const codeElement = children as React.ReactElement<{ className?: string; children?: string }>;
      const className = codeElement?.props?.className || '';
      const language = className.replace(/language-/, '');

      if (language === 'mermaid') {
        const code = codeElement?.props?.children || '';
        return <MermaidCodeBlock>{code}</MermaidCodeBlock>;
      }

      return <Pre {...props}>{children}</Pre>;
    },

    // Inline code
    code: ({ children, className, ...props }) => {
      // If it's inside a pre tag, let the Pre component handle it
      if (className?.includes('language-')) {
        return <code className={className} {...props}>{children}</code>;
      }
      return (
        <code
          className="px-1.5 py-0.5 bg-gray-100 text-indigo-600 rounded font-mono text-sm"
          {...props}
        >
          {children}
        </code>
      );
    },

    // Lists
    ul: ({ children, ...props }) => (
      <ul className="my-4 ml-6 list-disc space-y-2" {...props}>
        {children}
      </ul>
    ),
    ol: ({ children, ...props }) => (
      <ol className="my-4 ml-6 list-decimal space-y-2" {...props}>
        {children}
      </ol>
    ),
    li: ({ children, ...props }) => (
      <li className="leading-7 text-gray-700" {...props}>
        {children}
      </li>
    ),

    // Blockquotes
    blockquote: ({ children, ...props }) => (
      <blockquote
        className="my-4 pl-4 border-l-4 border-indigo-500 italic text-gray-600"
        {...props}
      >
        {children}
      </blockquote>
    ),

    // Tables
    table: ({ children, ...props }) => (
      <div className="my-6 overflow-x-auto">
        <table className="min-w-full border border-gray-200" {...props}>
          {children}
        </table>
      </div>
    ),
    thead: ({ children, ...props }) => (
      <thead className="bg-gray-50" {...props}>
        {children}
      </thead>
    ),
    th: ({ children, ...props }) => (
      <th
        className="px-4 py-3 text-left text-xs font-bold text-gray-600 uppercase tracking-wider border-b"
        {...props}
      >
        {children}
      </th>
    ),
    td: ({ children, ...props }) => (
      <td className="px-4 py-3 text-sm text-gray-700 border-b border-gray-100" {...props}>
        {children}
      </td>
    ),
    tr: ({ children, ...props }) => (
      <tr className="hover:bg-gray-50 transition-colors" {...props}>
        {children}
      </tr>
    ),

    // Horizontal rule
    hr: (props) => <hr className="my-8 border-gray-200" {...props} />,

    // Strong/Bold
    strong: ({ children, ...props }) => (
      <strong className="font-semibold text-gray-900" {...props}>
        {children}
      </strong>
    ),

    // Emphasis/Italic
    em: ({ children, ...props }) => (
      <em className="italic" {...props}>
        {children}
      </em>
    ),

    // Custom components
    CodeBlock,

    ...components,
  };
}

// Export the components for use in mdx-components.tsx at root
export const mdxComponents = useMDXComponents({});
