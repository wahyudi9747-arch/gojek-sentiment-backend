"""
BACKEND FASTAPI - Analisis Sentimen Ulasan Gojek
Menyajikan model terbaik (Model2_CNN_BiLSTM_Hybrid) hasil train_and_evaluate.py
ke aplikasi mobile Flutter.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model_service import sentiment_model
from . import database

app = FastAPI(title="Gojek Sentiment Analysis API")

# CORS dibuka agar bisa diakses dari emulator/HP fisik/web tanpa hambatan.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


@app.on_event("startup")
def on_startup():
    database.init_db()


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": sentiment_model.model_name,
        "classes": sentiment_model.class_names,
    }


@app.post("/api/sentiment/predict")
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Teks tidak boleh kosong")

    result = sentiment_model.predict(text)

    database.save_history(
        original_text=result["original_text"],
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
    )

    return result


@app.get("/api/sentiment/history")
def history(limit: int = 20):
    return database.get_history(limit=limit)
