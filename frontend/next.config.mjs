/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Rewrite /api/* to the FastAPI backend in dev; production uses the
  // absolute NEXT_PUBLIC_API_BASE_URL directly from lib/api.ts.
  async rewrites() {
    const backend = process.env.API_PROXY_TARGET || "http://localhost:8000";
    return [
      {
        source: "/api/backend/:path*",
        destination: `${backend}/:path*`,
      },
    ];
  },
};

export default nextConfig;
