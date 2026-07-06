import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "history.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_text TEXT,
            predicted_label TEXT,
            confidence REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_history(original_text: str, predicted_label: str, confidence: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO prediction_history (original_text, predicted_label, confidence, created_at) VALUES (?, ?, ?, ?)",
        (original_text, predicted_label, confidence, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
