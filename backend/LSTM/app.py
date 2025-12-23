import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# ---------------------------
# Streamlit page
# ---------------------------
st.set_page_config(page_title="Gold Price LSTM Demo", layout="wide")
st.title("🏅 Gold Price Prediction (LSTM) — Demo")
st.caption(
    "Upload CSV → clean/sort → backtest đúng nghĩa (test split) → forecast tương lai "
    "(có tuỳ chọn mode & clip để tránh drift)."
)


# ---------------------------
# Helpers
# ---------------------------
def load_artifacts(model_path: str, scaler_path: str):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def parse_and_clean(df: pd.DataFrame):
    # Clean col names
    df.columns = (
        df.columns.astype(str)
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    if "Date" not in df.columns:
        raise ValueError("CSV phải có cột 'Date'.")
    if "Price" not in df.columns:
        raise ValueError("CSV phải có cột 'Price'.")

    # Date: dataset của bạn dạng mm/dd/yyyy như 12/17/2025
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)

    # Price: xử lý dấu phẩy hàng nghìn và ký tự tiền tệ (nếu có)
    df["Price"] = (
        df["Price"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df.dropna(subset=["Date", "Price"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def make_sequences(series_scaled: np.ndarray, window: int):
    # series_scaled shape: (N, 1)
    X, y = [], []
    for i in range(window, len(series_scaled)):
        X.append(series_scaled[i - window : i, 0])
        y.append(series_scaled[i, 0])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)
    return X, y


def predict_series(model, scaler, prices: np.ndarray, window: int):
    """
    Predict on a contiguous price segment.
    Returns y_true, y_hat aligned to indices [window:].
    """
    scaled = scaler.transform(prices.reshape(-1, 1))
    X, y = make_sequences(scaled, window)

    if X.shape[0] == 0:
        raise ValueError(
            f"Không đủ dữ liệu để tạo sequence. Cần >= window+1. (window={window})"
        )

    yhat_scaled = model.predict(X, verbose=0).reshape(-1, 1)
    y_hat = scaler.inverse_transform(yhat_scaled).reshape(-1)
    y_true = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
    return y_true, y_hat


def forecast_recursive(model, scaler, prices: np.ndarray, window: int, steps: int, clip01: bool = True):
    """
    Recursive multi-step forecast (dễ drift nếu steps dài).
    clip01=True: giới hạn dự đoán trong [0,1] của MinMaxScaler để giảm drift cực đoan.
    """
    scaled = scaler.transform(prices.reshape(-1, 1)).reshape(-1)
    buf = scaled.tolist()

    preds_scaled = []
    for _ in range(steps):
        x = np.array(buf[-window:]).reshape(1, window, 1)
        p = float(model.predict(x, verbose=0).reshape(-1)[0])

        if clip01:
            p = float(np.clip(p, 0.0, 1.0))

        preds_scaled.append(p)
        buf.append(p)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).reshape(-1)
    return preds


def get_model_window_size(model):
    ish = model.input_shape
    if isinstance(ish, list):
        ish = ish[0]
    return int(ish[1])


# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("⚙️ Settings")

model_file = st.sidebar.text_input("Model path", value="lstm_gold.h5")
scaler_file = st.sidebar.text_input("Scaler path", value="scaler.pkl")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

test_ratio = st.sidebar.slider("Test ratio (for backtest)", 0.05, 0.30, 0.15, 0.05)

st.sidebar.divider()
st.sidebar.subheader("🔮 Forecast")
steps_ahead = st.sidebar.slider("Forecast days ahead", 1, 120, 14, 1)
forecast_mode = st.sidebar.selectbox("Forecast mode", ["Recursive (multi-step)", "One-step only (next day)"])
use_clip = st.sidebar.checkbox("Clip to scaler range [0,1] (reduce drift)", value=True)
use_bdays = st.sidebar.checkbox("Use business days (B) for future dates", value=True)

st.sidebar.divider()
run_btn = st.sidebar.button("🚀 Run Prediction")


# ---------------------------
# Main
# ---------------------------
if not uploaded:
    st.info("⬅️ Upload CSV ở sidebar để bắt đầu.")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded)
    df = parse_and_clean(df_raw)
except Exception as e:
    st.error(f"Lỗi đọc/clean CSV: {e}")
    st.stop()

# Basic info
st.subheader("📄 Data Preview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Date min", str(df["Date"].min().date()))
c3.metric("Date max", str(df["Date"].max().date()))
st.dataframe(df.head(10), use_container_width=True)

# Plot raw
st.subheader("📈 Price History (sorted ascending)")
fig0 = plt.figure()
plt.plot(df["Date"], df["Price"])
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
st.pyplot(fig0)

if not run_btn:
    st.stop()

# Load model/scaler
try:
    model, scaler = load_artifacts(model_file, scaler_file)
    window = get_model_window_size(model)
except Exception as e:
    st.error(
        f"Lỗi load model/scaler: {e}\n\n"
        f"Đảm bảo bạn có '{model_file}' và '{scaler_file}' đúng đường dẫn."
    )
    st.stop()

st.success(f"✅ Loaded model & scaler. Model window_size = {window}")

prices = df["Price"].values.astype(float)
N = len(prices)

# Sanity check
if N <= window + 5:
    st.error(f"Dữ liệu quá ít. Cần ít nhất window+5 dòng. (N={N}, window={window})")
    st.stop()

# ---------------------------
# Backtest (đúng nghĩa theo test split)
# ---------------------------
st.subheader("✅ Backtest (Test split)")

test_size = int(N * test_ratio)
test_size = max(test_size, 30)  # tối thiểu để nhìn chart
test_size = min(test_size, N - window - 1)  # tránh vượt
if test_size <= 0:
    st.error("Test size không hợp lệ. Giảm window hoặc tăng số dòng dữ liệu.")
    st.stop()

# Lấy 1 đoạn đủ để tạo sequence cho test: (window + test_size)
segment = prices[-(window + test_size):]
dates_test = df["Date"].iloc[-test_size:].reset_index(drop=True)

try:
    y_true_t, y_hat_t = predict_series(model, scaler, segment, window)
    # y_true_t/y_hat_t length == test_size
except Exception as e:
    st.error(f"Lỗi backtest predict: {e}")
    st.stop()

fig1 = plt.figure()
plt.plot(dates_test, y_true_t, label="Actual (test)")
plt.plot(dates_test, y_hat_t, label="Predicted (test)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
st.pyplot(fig1)

mae = float(np.mean(np.abs(y_true_t - y_hat_t)))
rmse = float(np.sqrt(np.mean((y_true_t - y_hat_t) ** 2)))
st.write(f"**MAE (test):** {mae:,.2f} | **RMSE (test):** {rmse:,.2f}")

# ---------------------------
# Forecast (thực tế: chỉ đáng tin ngắn hạn)
# ---------------------------
st.subheader("🔮 Forecast (practical)")

freq = "B" if use_bdays else "D"
last_date = df["Date"].iloc[-1]

if forecast_mode == "One-step only (next day)":
    steps = 1
else:
    steps = steps_ahead

future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq=freq)

try:
    future_preds = forecast_recursive(model, scaler, prices, window, steps, clip01=use_clip)
except Exception as e:
    st.error(f"Lỗi forecast: {e}")
    st.stop()

# Plot recent + forecast
lookback = min(300, N)
fig2 = plt.figure()
plt.plot(df["Date"].tail(lookback), df["Price"].tail(lookback), label="Recent Actual")
plt.plot(future_dates, future_preds, label="Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
st.pyplot(fig2)

# Output table
out = pd.DataFrame({"Date": future_dates, "ForecastPrice": future_preds})
st.dataframe(out, use_container_width=True)

st.download_button(
    "⬇️ Download forecast CSV",
    out.to_csv(index=False).encode("utf-8-sig"),
    file_name="forecast.csv",
    mime="text/csv",
)

st.caption(
    "⚠️ Lưu ý: Multi-step recursive forecast dễ drift (đặc biệt khi dữ liệu cuối chuỗi tăng/giảm mạnh). "
    "Demo thực tế nên để 7–14 ngày hoặc dùng mode 'One-step'."
)
