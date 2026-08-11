/**
 * Centralised access to the two public env vars the frontend reads.
 * Never import process.env directly outside this file.
 */

export const API_BASE_URL =
  (process.env.NEXT_PUBLIC_API_BASE_URL || "/api/backend").replace(/\/$/, "");

export const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";
