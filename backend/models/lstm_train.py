import os

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from utils.metrics import mape
from utils.preprocess import prepare_dataframe

MODEL_PATH = "artifacts/lstm_gold.h5"
SCALER_PATH = "artifacts/scaler.pkl"


def make_sequences(values_scaled: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(window_size, len(values_scaled)):
        X.append(values_scaled[i - window_size:i, 0])
        y.append(values_scaled[i, 0])
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


def train_lstm(
    df,
    date_col: str,
    price_col: str,
    window_size: int,
    test_year: int,
    epochs: int,
    batch_size: int,
    lstm_units: int,
    dropout: float,
    learning_rate: float,
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
    if len(test_df) < 5:
        raise ValueError(
            f"Invalid input: test set too small ({len(test_df)} rows) for year {test_year}."
        )

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[[price_col]].values)
    test_scaled = scaler.transform(test_df[[price_col]].values)

    X_train, y_train = make_sequences(train_scaled, window_size)

    combined = np.vstack([train_scaled[-window_size:], test_scaled])
    X_val, y_val = make_sequences(combined, window_size)

    if len(X_train) == 0:
        raise ValueError(
            "Invalid input: not enough data after scaling to build training sequences."
        )
    if len(X_val) == 0:
        raise ValueError(
            "Invalid input: not enough data to build validation sequences."
        )

    model = Sequential(
        [
            LSTM(lstm_units, input_shape=(window_size, 1)),
            Dropout(dropout),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    y_pred_scaled = model.predict(X_val, verbose=0)
    y_true = scaler.inverse_transform(y_val).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    loss = float(history.history["loss"][-1]) if history.history.get("loss") else None
    val_loss = (
        float(history.history["val_loss"][-1])
        if history.history.get("val_loss")
        else None
    )

    return {
        "loss": loss,
        "val_loss": val_loss,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "mape": mape(y_true, y_pred),
    }
