"""
MODEL SERVICE
Load model + tokenizer + label encoder sekali saat startup (bukan per-request),
supaya prediksi cepat. Menggunakan Model2_CNN_BiLSTM_Hybrid (model terbaik
berdasarkan f1-score, lihat model_info.json).
"""
import os
import json
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .preprocessing import preprocess_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class SentimentModel:
    def __init__(self):
        with open(os.path.join(MODEL_DIR, "model_info.json")) as f:
            info = json.load(f)
        self.model_name = info["best_model"]
        self.max_len = info["max_len"]

        self.model = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "best_model.keras")
        )
        with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
            self.tokenizer = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)

        self.class_names = list(self.label_encoder.classes_)

        # warm-up: hindari cold-start lambat di request pertama
        self._predict_raw("aplikasi bagus")

    def _predict_raw(self, clean_text: str):
        seq = self.tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(seq, maxlen=self.max_len, padding="post", truncating="post")
        probs = self.model.predict(padded, verbose=0)[0]
        return probs

    def predict(self, original_text: str) -> dict:
        clean = preprocess_text(original_text)
        probs = self._predict_raw(clean)

        predicted_idx = int(np.argmax(probs))
        predicted_label = self.class_names[predicted_idx]
        confidence = float(probs[predicted_idx])
        probabilities = {name: float(p) for name, p in zip(self.class_names, probs)}

        return {
            "original_text": original_text,
            "clean_text": clean,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": probabilities,
        }


# instance singleton, di-load sekali saat modul pertama kali diimport
sentiment_model = SentimentModel()
