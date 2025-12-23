// trainService.ts

// Read env var (Vercel: set VITE_API_BASE_URL = https://predict-gold-backend.onrender.com)
const raw = import.meta.env.VITE_API_BASE_URL as string | undefined;
const envApiBase = raw?.trim().replace(/\/+$/, ""); // remove trailing slash

/**
 * DEV (local): use Vite proxy (/api) when env is missing/empty
 * PROD (deploy): require VITE_API_BASE_URL
 */
const API_BASE =
  envApiBase && envApiBase.length > 0
    ? envApiBase
    : import.meta.env.DEV
      ? "/api"
      : "";

/**
 * Ensure API base exists in production.
 * Also returns base without trailing slash to avoid //train/lstm.
 */
function requireApiBase() {
  if (!API_BASE) {
    throw new Error(
      "Missing VITE_API_BASE_URL in production. Please set it in Vercel Environment Variables."
    );
  }
  return API_BASE.replace(/\/+$/, "");
}

async function parseError(res: Response) {
  // Best-effort parsing for FastAPI errors (may return JSON or plain text)
  const contentType = res.headers.get("content-type") || "";
  try {
    if (contentType.includes("application/json")) {
      const data = await res.json();
      if (typeof data === "string") {
        return data;
      }
      if (data && typeof data === "object" && "detail" in data) {
        const detail = (data as { detail?: unknown }).detail;
        if (typeof detail === "string") {
          return detail;
        }
        return JSON.stringify(detail);
      }
      return JSON.stringify(data);
    }
    return await res.text();
  } catch {
    return "";
  }
}

export async function startLSTMTrainJob(params: {
  file: File;
  dateCol: string;
  priceCol: string;
  windowSize: number;
  signal?: AbortSignal;
}) {
  const base = requireApiBase();

  const form = new FormData();
  form.append("file", params.file);
  form.append("date_col", params.dateCol);
  form.append("price_col", params.priceCol);
  form.append("window_size", String(params.windowSize));

  const res = await fetch(`${base}/train/lstm`, {
    method: "POST",
    body: form,
    signal: params.signal,
  });

  if (!res.ok) {
    const detail = await parseError(res);
    throw new Error(detail ? `HTTP ${res.status}: ${detail}` : `HTTP ${res.status}`);
  }

  return res.json() as Promise<{ job_id: string }>;
}

export async function getLSTMTrainJobStatus(jobId: string, signal?: AbortSignal) {
  const base = requireApiBase();
  const res = await fetch(`${base}/train/lstm/status/${jobId}`, {
    method: "GET",
    signal,
  });

  if (!res.ok) {
    const detail = await parseError(res);
    throw new Error(detail ? `HTTP ${res.status}: ${detail}` : `HTTP ${res.status}`);
  }

  return res.json() as Promise<{
    job_id: string;
    status: "pending" | "running" | "done" | "error";
    result?: Record<string, unknown>;
    error?: string;
  }>;
}
