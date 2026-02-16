import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Documentation - Entity Database',
  description: 'Documentation for the corp-entity-db entity database library',
};

export default function DocsLayout({ children }: { children: React.ReactNode }) {
  return children;
}
