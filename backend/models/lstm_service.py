import numpy as np
import joblib
from tensorflow.keras.models import load_model
from utils.preprocess import prepare_dataframe
from utils.metrics import mape

MODEL = load_model("artifacts/lstm_gold.h5")
SCALER = joblib.load("artifacts/scaler.pkl")

def make_sequences(values_scaled, window_size):
    X, y = [], []
    for i in range(window_size, len(values_scaled)):
        X.append(values_scaled[i-window_size:i, 0])
        y.append(values_scaled[i, 0])
    return np.array(X).reshape(-1, window_size, 1), np.array(y).reshape(-1, 1)

def predict_lstm(df, date_col, price_col, window_size, test_year):
    df = prepare_dataframe(df, date_col, price_col)

    is_test = df[date_col].dt.year == test_year
    train_df = df.loc[~is_test]
    test_df  = df.loc[is_test]

    train_scaled = SCALER.transform(train_df[[price_col]])
    test_scaled  = SCALER.transform(test_df[[price_col]])

    X_test, y_test = make_sequences(test_scaled, window_size)
    y_pred_scaled = MODEL.predict(X_test, verbose=0)

    y_true = SCALER.inverse_transform(y_test).flatten()
    y_pred = SCALER.inverse_transform(y_pred_scaled).flatten()

    dates = test_df[date_col].iloc[window_size:].astype(str).tolist()

    return {
        "dates": dates,
        "actual": y_true.tolist(),
        "predicted": y_pred.tolist(),
        "mape": mape(y_true, y_pred)
    }
