from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

from models.lstm_service import predict_lstm

app = FastAPI()

# ================= CORS CONFIG =================
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")

if FRONTEND_ORIGIN is None:
    print("FRONTEND_ORIGIN not set, allowing localhost for development")
    origins = [
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # fallback nếu bạn từng dùng
    ]
else:
    origins = [o.strip() for o in FRONTEND_ORIGIN.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= API =================
@app.post("/predict/lstm")
async def predict_lstm_api(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    price_col: str = Form(...),
    window_size: int = Form(60),
    test_year: int = Form(2022),
):
    df = pd.read_csv(file.file)
    return predict_lstm(df, date_col, price_col, window_size, test_year)
