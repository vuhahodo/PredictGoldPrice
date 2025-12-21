// predictService.ts

const raw = import.meta.env.VITE_API_BASE_URL as string | undefined;
const envApiBase = raw?.trim();

// DEV (local): use Vite proxy (/api) when env is missing/empty
// PROD (deploy): require VITE_API_BASE_URL
const API_BASE =
  envApiBase && envApiBase.length > 0
    ? envApiBase
    : import.meta.env.DEV
      ? "/api"
      : "";

function requireApiBase() {
  if (!API_BASE) {
    throw new Error(
      "Missing VITE_API_BASE_URL in production. Please set it in Vercel Environment Variables."
    );
  }
  return API_BASE;
}

export async function predictLSTMFromBackend(params: {
  file: File;
  dateCol: string;
  priceCol: string;
  windowSize: number;
  testYear: number;
}) {
  const base = requireApiBase();

  const form = new FormData();
  form.append("file", params.file);
  form.append("date_col", params.dateCol);
  form.append("price_col", params.priceCol);
  form.append("window_size", String(params.windowSize));
  form.append("test_year", String(params.testYear));

  const res = await fetch(`${base}/predict/lstm`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${txt}`);
  }

  return res.json();
}
