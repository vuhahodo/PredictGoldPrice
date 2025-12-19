const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function predictLSTMFromBackend(params: {
  file: File;
  dateCol: string;
  priceCol: string;
  windowSize: number;
  testYear: number;
}) {
  const form = new FormData();
  form.append("file", params.file);
  form.append("date_col", params.dateCol);
  form.append("price_col", params.priceCol);
  form.append("window_size", String(params.windowSize));
  form.append("test_year", String(params.testYear));

  const res = await fetch(`${API_BASE}/predict/lstm`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HTTP ${res.status}: ${txt}`);
  }

  return res.json();
}
