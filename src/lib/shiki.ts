import { createHighlighter, type Highlighter, type BundledLanguage, type BundledTheme } from 'shiki';

// Singleton highlighter instance
let highlighter: Highlighter | null = null;

// Languages we need for documentation
const LANGUAGES: BundledLanguage[] = ['python', 'typescript', 'bash', 'xml', 'json', 'javascript', 'sql'];

// Custom theme matching our brutalist design (dark code blocks)
const THEME: BundledTheme = 'github-dark';

/**
 * Get or create the Shiki highlighter instance.
 * Uses singleton pattern to avoid re-creating the highlighter.
 */
export async function getHighlighter(): Promise<Highlighter> {
  if (!highlighter) {
    highlighter = await createHighlighter({
      themes: [THEME],
      langs: LANGUAGES,
    });
  }
  return highlighter;
}

/**
 * Highlight code with Shiki.
 * Returns HTML string with syntax highlighting.
 */
export async function highlightCode(
  code: string,
  language: BundledLanguage
): Promise<string> {
  const h = await getHighlighter();
  return h.codeToHtml(code, {
    lang: language,
    theme: THEME,
  });
}

/**
 * Supported languages for documentation
 */
export type SupportedLanguage = 'python' | 'typescript' | 'bash' | 'xml' | 'json' | 'javascript' | 'sql';

/**
 * Check if a language is supported
 */
export function isLanguageSupported(lang: string): lang is SupportedLanguage {
  return LANGUAGES.includes(lang as BundledLanguage);
}
