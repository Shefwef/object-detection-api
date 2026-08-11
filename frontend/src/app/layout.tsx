import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: "Object Detection Studio",
  description:
    "Unified interface for YOLOv8, Detectron2, Grounding DINO and SAM — powered by a single FastAPI backend.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <Nav />
        <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pb-24 pt-6">{children}</main>
        <footer className="border-t border-surface-border/60 py-6 text-center text-xs text-slate-500">
          Built with FastAPI, Next.js and open-weight CV models · {new Date().getFullYear()}
        </footer>
      </body>
    </html>
  );
}
