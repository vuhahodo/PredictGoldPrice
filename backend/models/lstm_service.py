import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
from utils.preprocess import prepare_dataframe
from utils.metrics import mape

MODEL_PATH = "artifacts/lstm_gold.h5"
SCALER_PATH = "artifacts/scaler.pkl"
MODEL_META_PATH = "artifacts/model_meta.json"
MODEL = None
SCALER = None


def load_artifacts():
    global MODEL, SCALER
    if MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
        MODEL = load_model(MODEL_PATH)
    if SCALER is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Model artifact not found: {SCALER_PATH}")
        SCALER = joblib.load(SCALER_PATH)
    return MODEL, SCALER


def make_sequences(values_scaled: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(window_size, len(values_scaled)):
        X.append(values_scaled[i - window_size:i, 0])
        y.append(values_scaled[i, 0])
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


def load_model_metadata():
    if not os.path.exists(MODEL_META_PATH):
        return None
    with open(MODEL_META_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_model_metadata(metadata: dict):
    os.makedirs(os.path.dirname(MODEL_META_PATH), exist_ok=True)
    with open(MODEL_META_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def build_lstm_model(window_size: int):
    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
            LSTM(50),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_lstm(
    df,
    date_col: str,
    price_col: str,
    window_size: int,
    test_year: int,
    epochs: int = 10,
    batch_size: int = 32,
):
    if date_col not in df.columns:
        raise ValueError(f"Invalid input: missing column '{date_col}'.")
    if price_col not in df.columns:
        raise ValueError(f"Invalid input: missing column '{price_col}'.")

    df = prepare_dataframe(df, date_col, price_col)

    is_test = df[date_col].dt.year == int(test_year)
    train_df = df.loc[~is_test].reset_index(drop=True)
    test_df = df.loc[is_test].reset_index(drop=True)

    if len(train_df) < window_size:
        raise ValueError(
            f"Invalid input: train set too small ({len(train_df)} rows) for window size {window_size}."
        )
    if len(test_df) == 0:
        raise ValueError(f"Invalid input: no data for test year {test_year}.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df[[price_col]].values)

    X_train, y_train = make_sequences(train_scaled, window_size)

    model = build_lstm_model(window_size)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    global MODEL, SCALER
    MODEL = model
    SCALER = scaler

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(df)),
        "window_size": int(window_size),
        "test_year": int(test_year),
        "columns": {
            "date": date_col,
            "price": price_col,
        },
    }
    save_model_metadata(metadata)

    return {
        "message": "LSTM model trained successfully.",
        "metadata": metadata,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
    }


def predict_lstm(df, date_col: str, price_col: str, window_size: int, test_year: int):
    if date_col not in df.columns:
        raise ValueError(f"Invalid input: missing column '{date_col}'.")
    if price_col not in df.columns:
        raise ValueError(f"Invalid input: missing column '{price_col}'.")

    df = prepare_dataframe(df, date_col, price_col)

    is_test = df[date_col].dt.year == int(test_year)
    train_df = df.loc[~is_test].reset_index(drop=True)
    test_df  = df.loc[is_test].reset_index(drop=True)

    if len(train_df) < window_size:
        raise ValueError(
            f"Invalid input: train set too small ({len(train_df)} rows) for window size {window_size}."
        )
    if len(test_df) == 0:
        raise ValueError(f"Invalid input: no data for test year {test_year}.")
    if len(test_df) < 5:
        raise ValueError(
            f"Invalid input: test set too small ({len(test_df)} rows) for year {test_year}."
        )

    model, scaler = load_artifacts()

    # Scale using pretrained scaler
    train_scaled = scaler.transform(train_df[[price_col]].values)
    test_scaled  = scaler.transform(test_df[[price_col]].values)

    # ✅ Correct alignment: include last window from train before test
    combined = np.vstack([train_scaled[-window_size:], test_scaled])

    X_test, y_test = make_sequences(combined, window_size)
    y_pred_scaled = model.predict(X_test, verbose=0)

    y_true = scaler.inverse_transform(y_test).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

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
        "metadata": load_model_metadata(),
    }
