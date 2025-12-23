from pathlib import Path
import numpy as np
import joblib
import os
import threading
import logging

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

from utils.preprocess import prepare_dataframe
from utils.metrics import mape

# =====================================================
# PATHS
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
# Point to the LSTM training directory instead
ARTIFACT_DIR = BASE_DIR.parent / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "lstm_gold.h5"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"

# =====================================================
# GLOBAL ARTIFACTS (❗ BẮT BUỘC)
# =====================================================
MODEL = None
SCALER = None

ARTIFACT_LOCK = threading.RLock()
logger = logging.getLogger(__name__)

# =====================================================
# HELPERS
# =====================================================
def get_model_window_size(model):
    ish = model.input_shape
    if isinstance(ish, list):
        ish = ish[0]
    if ish is None or len(ish) < 2 or ish[1] is None:
        raise ValueError("Invalid model input shape")
    return int(ish[1])


def make_sequences(values_scaled: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(window_size, len(values_scaled)):
        X.append(values_scaled[i - window_size:i, 0])
        y.append(values_scaled[i, 0])
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


# =====================================================
# LOAD ARTIFACTS (MODEL + SCALER)
# =====================================================
def load_artifacts():
    global MODEL, SCALER

    with ARTIFACT_LOCK:
        if MODEL is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
            MODEL = load_model(str(MODEL_PATH), compile=False)

        if SCALER is None:
            if not SCALER_PATH.exists():
                raise FileNotFoundError(f"Missing scaler: {SCALER_PATH}")
            SCALER = joblib.load(str(SCALER_PATH))

        return MODEL, SCALER


# =====================================================
# TRAIN (OPTIONAL)
# =====================================================
def train_lstm(
    df,
    date_col: str,
    price_col: str,
    window_size: int,
    epochs: int = 20,
    batch_size: int = 32,
):
    df = prepare_dataframe(df, date_col, price_col)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[[price_col]].values)

    X_train, y_train = make_sequences(scaled, window_size)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
        LSTM(64),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    global MODEL, SCALER
    MODEL = model
    SCALER = scaler


# =====================================================
# PREDICT API
# =====================================================
def predict_lstm(df, date_col: str, price_col: str, window_size: int, test_year: int):
    try:
        logger.info(f"Starting LSTM prediction for test_year={test_year}, window_size={window_size}")
        
        df = prepare_dataframe(df, date_col, price_col)
        logger.info(f"Prepared dataframe: {len(df)} rows")

        is_test = df[date_col].dt.year == int(test_year)
        train_df = df.loc[~is_test].reset_index(drop=True)
        test_df = df.loc[is_test].reset_index(drop=True)

        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        if len(test_df) == 0:
            raise ValueError(f"No data for test year {test_year}")

        model, scaler = load_artifacts()
        logger.info("Artifacts loaded successfully")

        model_window = get_model_window_size(model)
        logger.info(f"Model expects window_size={model_window}")
        
        if window_size != model_window:
            logger.warning(
                f"Window mismatch: input={window_size}, model={model_window}. Using model window."
            )
            window_size = model_window

        # Check if we have enough training data
        if len(train_df) < window_size:
            raise ValueError(f"Not enough training data: need {window_size}, got {len(train_df)}")

        train_scaled = scaler.transform(train_df[[price_col]].values)
        test_scaled = scaler.transform(test_df[[price_col]].values)

        combined = np.vstack([train_scaled[-window_size:], test_scaled])
        logger.info(f"Combined data shape: {combined.shape}")

        X_test, y_test = make_sequences(combined, window_size)
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        if len(X_test) == 0:
            raise ValueError(f"No test sequences created. Need at least {window_size} combined data points.")

        y_pred_scaled = model.predict(X_test, verbose=0)
        logger.info(f"Prediction completed: {y_pred_scaled.shape}")

        y_true = scaler.inverse_transform(y_test).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

        dates = test_df[date_col].iloc[:len(y_true)].astype(str).tolist()

        mape_value = mape(y_true, y_pred)
        logger.info(f"MAPE calculated: {mape_value}")

        return {
            "dates": dates,
            "actual": y_true.tolist(),
            "predicted": y_pred.tolist(),
            "mape": mape_value,
            "test_year": int(test_year),
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
            "window_size": int(window_size),
        }
    except Exception as e:
        logger.error(f"Error in predict_lstm: {str(e)}", exc_info=True)
        raise
