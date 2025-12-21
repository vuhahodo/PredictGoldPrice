// predictService.ts

// Read env var (Vercel: set VITE_API_BASE_URL = https://predict-gold-backend.onrender.com)
const raw = import.meta.env.VITE_API_BASE_URL as string | undefined;
const envApiBase = raw?.trim().replace(/\/+$/, ""); // ✅ remove trailing slash

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
 * Also returns base without trailing slash to avoid //predict/lstm.
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
      return JSON.stringify(data);
    }
    return await res.text();
  } catch {
    return "";
  }
}

export async function predictLSTMFromBackend(params: {
  file: File;
  dateCol: string;
  priceCol: string;
  windowSize: number;
  testYear: number;
  // Optional: AbortController signal if you want to cancel request from UI
  signal?: AbortSignal;
}) {
  const base = requireApiBase();

  const form = new FormData();
  // ✅ keys must match FastAPI: file, date_col, price_col, window_size, test_year
  form.append("file", params.file);
  form.append("date_col", params.dateCol);
  form.append("price_col", params.priceCol);
  form.append("window_size", String(params.windowSize));
  form.append("test_year", String(params.testYear));

  const res = await fetch(`${base}/predict/lstm`, {
    method: "POST",
    body: form,
    signal: params.signal,
    // ✅ IMPORTANT: DO NOT set "Content-Type" manually for FormData.
    // Browser will set proper multipart boundary automatically.
  });

  if (!res.ok) {
    const detail = await parseError(res);
    throw new Error(detail ? `HTTP ${res.status}: ${detail}` : `HTTP ${res.status}`);
  }

  return res.json();
}
