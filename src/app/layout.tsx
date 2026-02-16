import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Entity Database Search | corp-o-rate",
  description: "Search organizations, people, roles, and locations across 9.7M+ organizations and 63M+ people.",
  keywords: ["entity search", "organizations", "people", "NER", "knowledge base", "corp-o-rate"],
  authors: [{ name: "corp-o-rate" }],
  openGraph: {
    title: "Entity Database Search | corp-o-rate",
    description: "Search organizations, people, roles, and locations.",
    siteName: "Entity Database Search",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen`}
      >
        {children}
      </body>
    </html>
  );
}
