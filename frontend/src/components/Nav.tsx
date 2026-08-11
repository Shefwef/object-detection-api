"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Detect" },
  { href: "/compare", label: "Compare" },
  { href: "/metrics", label: "Metrics" },
];

export default function Nav() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-30 border-b border-surface-border/60 bg-surface/70 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-center gap-2">
          <span className="grid h-8 w-8 place-items-center rounded-md bg-accent/20 text-accent">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <path d="M9 9h6v6H9z" />
            </svg>
          </span>
          <span className="text-sm font-semibold tracking-wide text-white">
            Object Detection Studio
          </span>
        </Link>

        <nav className="flex items-center gap-1 text-sm">
          {links.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded-lg px-3 py-1.5 transition ${
                  active
                    ? "bg-white/10 text-white"
                    : "text-slate-300 hover:bg-white/5 hover:text-white"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
          <a
            href="https://github.com/Shefwef/object-detection-api"
            target="_blank"
            rel="noreferrer"
            className="btn-ghost text-xs"
          >
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}
