from fastapi import BackgroundTasks, FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import pandas as pd
import os

from models.lstm_service import predict_lstm
from models.lstm_train import train_lstm
from utils.job_store import JobStore

app = FastAPI()
job_store = JobStore()

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
    try:
        df = pd.read_csv(file.file, usecols=[date_col, price_col])
        return predict_lstm(df, date_col, price_col, window_size, test_year)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unexpected server error while running prediction.",
        ) from exc


@app.post("/train/lstm")
async def train_lstm_api(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    date_col: str = Form(...),
    price_col: str = Form(...),
    window_size: int = Form(60),
    test_year: int = Form(2022),
    epochs: int = Form(20),
    batch_size: int = Form(32),
    lstm_units: int = Form(50),
    dropout: float = Form(0.2),
    learning_rate: float = Form(0.001),
):
    try:
        contents = await file.read()
        job_id = job_store.create()

        def run_training_job() -> None:
            job_store.set_running(job_id)
            try:
                df = pd.read_csv(BytesIO(contents), usecols=[date_col, price_col])
                result = train_lstm(
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
                job_store.set_done(job_id, result)
            except ValueError as exc:
                job_store.set_error(job_id, str(exc))
            except Exception:
                job_store.set_error(
                    job_id,
                    "Unexpected server error while training model.",
                )

        background_tasks.add_task(run_training_job)
        return {"job_id": job_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unexpected server error while training model.",
        ) from exc


@app.get("/train/lstm/status/{job_id}")
async def train_lstm_status(job_id: str):
    record = job_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    payload = {"job_id": job_id, "status": record.status}
    if record.status == "done":
        payload["result"] = record.result
    if record.status == "error":
        payload["error"] = record.error
    return payload
