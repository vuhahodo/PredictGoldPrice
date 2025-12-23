import numpy as np
import joblib
import os
import threading
import logging
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from utils.preprocess import prepare_dataframe
from utils.metrics import mape

MODEL_PATH = "artifacts/lstm_gold.h5"
SCALER_PATH = "artifacts/scaler.pkl"
MODEL = None
SCALER = None
ARTIFACT_LOCK = threading.RLock()
logger = logging.getLogger(__name__)


def get_model_window_size(model):
    """Extract window size from model's input shape."""
    try:
        ish = model.input_shape
        if isinstance(ish, list):
            ish = ish[0]
        if ish is None or len(ish) < 2 or ish[1] is None:
            raise ValueError("Model input shape is invalid or not fully defined")
        return int(ish[1])
    except (AttributeError, IndexError, TypeError) as e:
        raise ValueError(f"Failed to extract window size from model: {e}") from e


def load_artifacts():
    global MODEL, SCALER
    with ARTIFACT_LOCK:
        if MODEL is None:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
            MODEL = load_model(MODEL_PATH)
        if SCALER is None:
            if not os.path.exists(SCALER_PATH):
                raise FileNotFoundError(f"Model artifact not found: {SCALER_PATH}")
            SCALER = joblib.load(SCALER_PATH)
        return MODEL, SCALER


def _set_artifacts(model, scaler):
    global MODEL, SCALER
    with ARTIFACT_LOCK:
        MODEL = model
        SCALER = scaler


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
    epochs: int = 20,
    batch_size: int = 32,
):
    if date_col not in df.columns:
        raise ValueError(f"Invalid input: missing column '{date_col}'.")
    if price_col not in df.columns:
        raise ValueError(f"Invalid input: missing column '{price_col}'.")

    df = prepare_dataframe(df, date_col, price_col)

    if len(df) < window_size + 1:
        raise ValueError(
            f"Invalid input: dataset too small ({len(df)} rows) for window size {window_size}."
        )

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[[price_col]].values)
    X_train, y_train = make_sequences(scaled, window_size)

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
            LSTM(64),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    _set_artifacts(model, scaler)

    last_loss = float(history.history["loss"][-1]) if history.history.get("loss") else None
    return {
        "train_size": int(len(df)),
        "window_size": int(window_size),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "loss": last_loss,
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

    model, scaler = load_artifacts()
    
    # ✅ Auto-detect window size from model if not properly set
    model_window_size = get_model_window_size(model)
    if window_size != model_window_size:
        logger.warning(
            f"Window size mismatch: provided={window_size}, model expects={model_window_size}. "
            f"Auto-correcting to use model's window size: {model_window_size}"
        )
        window_size = model_window_size

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
    }
