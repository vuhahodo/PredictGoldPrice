import numpy as np
import joblib
from tensorflow.keras.models import load_model  # type: ignore
from utils.preprocess import prepare_dataframe
from utils.metrics import mape

MODEL = load_model("artifacts/lstm_gold.h5")
SCALER = joblib.load("artifacts/scaler.pkl")


def make_sequences(values_scaled: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(window_size, len(values_scaled)):
        X.append(values_scaled[i - window_size:i, 0])
        y.append(values_scaled[i, 0])
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


def predict_lstm(df, date_col: str, price_col: str, window_size: int, test_year: int):
    df = prepare_dataframe(df, date_col, price_col)

    is_test = df[date_col].dt.year == int(test_year)
    train_df = df.loc[~is_test].reset_index(drop=True)
    test_df  = df.loc[is_test].reset_index(drop=True)

    if len(train_df) < window_size:
        raise ValueError(f"Train too small: {len(train_df)} rows < window_size={window_size}")
    if len(test_df) < 5:
        raise ValueError(f"Test too small: {len(test_df)} rows in year {test_year}")

    # Scale using pretrained scaler
    train_scaled = SCALER.transform(train_df[[price_col]].values)
    test_scaled  = SCALER.transform(test_df[[price_col]].values)

    # ✅ Correct alignment: include last window from train before test
    combined = np.vstack([train_scaled[-window_size:], test_scaled])

    X_test, y_test = make_sequences(combined, window_size)
    y_pred_scaled = MODEL.predict(X_test, verbose=0)

    y_true = SCALER.inverse_transform(y_test).flatten()
    y_pred = SCALER.inverse_transform(y_pred_scaled).flatten()

    # ✅ dates aligned with y_true/y_pred: start from first day of test
    dates = test_df[date_col].iloc[:len(y_true)].astype(str).tolist()

    return {
        "dates": dates,
        "actual": y_true.tolist(),
        "predicted": y_pred.tolist(),
        "mape": mape(y_true, y_pred),
        # (optional) helpful for frontend labeling
        "test_year": int(test_year),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
    }
