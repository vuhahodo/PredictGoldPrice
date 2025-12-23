# backend/main.py

import os
import logging
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.lstm_service import predict_lstm
from models.lstm_train import train_lstm

app = FastAPI()
logger = logging.getLogger(__name__)

# ================= CORS CONFIG =================
# FRONTEND_ORIGIN can be a comma-separated list, e.g.
# FRONTEND_ORIGIN="https://predict-gold-price.vercel.app,http://localhost:5173"
raw_frontend_origin = os.getenv("FRONTEND_ORIGIN")

if not raw_frontend_origin:
    logger.warning("FRONTEND_ORIGIN not set, allowing localhost for development")
    origins = [
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # fallback
    ]
else:
    # ✅ Normalize: trim spaces + remove trailing slashes to avoid CORS mismatch
    origins = [o.strip().rstrip("/") for o in raw_frontend_origin.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= HELPERS =================
def read_uploaded_csv(file: UploadFile, sep: Optional[str]) -> pd.DataFrame:
    """
    Read uploaded CSV with UTF-8 BOM support.
    Optional separator (sep) for cases like ';' or '\t'.
    """
    try:
        if sep:
            return pd.read_csv(file.file, encoding="utf-8-sig", sep=sep)
        return pd.read_csv(file.file, encoding="utf-8-sig")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid CSV or encoding") from exc


# ================= API =================
@app.post("/predict/lstm")
async def predict_lstm_api(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    price_col: str = Form(...),
    window_size: int = Form(180),  # Will auto-detect from model if mismatch
    test_year: int = Form(2022),
    sep: Optional[str] = Form(None),
):
    try:
        df = read_uploaded_csv(file, sep)
        return predict_lstm(df, date_col, price_col, window_size, test_year)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        # e.g. missing model artifact
        logger.error("Prediction failed due to missing artifact: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected server error while running prediction: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Unexpected server error while running prediction.",
        ) from exc


@app.post("/train/lstm")
async def train_lstm_api(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    price_col: str = Form(...),
    window_size: int = Form(60),  # For training a new model
    test_year: int = Form(2022),
    epochs: int = Form(20),
    batch_size: int = Form(32),
    lstm_units: int = Form(50),
    dropout: float = Form(0.2),
    learning_rate: float = Form(0.001),
    sep: Optional[str] = Form(None),
):
    try:
        df = read_uploaded_csv(file, sep)
        return train_lstm(
            df,
            date_col,
            price_col,
            window_size,
            test_year,
            epochs,
            batch_size,
            lstm_units,
            dropout,
            learning_rate,
        )
    except ValueError as exc:
        logger.warning("Train LSTM failed due to invalid input: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.error("Train LSTM failed due to missing file: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except OSError as exc:
        logger.error("Train LSTM failed to save model artifacts: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save model artifacts: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while training model: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Unexpected server error while training model.",
        ) from exc


# (Optional) health endpoint - doesn't remove any existing functionality.
# Useful to quickly verify Render is up without using /docs.
@app.get("/health")
def health():
    return {"status": "ok"}
