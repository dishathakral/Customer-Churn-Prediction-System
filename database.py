import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if os.environ.get("VERCEL"):
    DB_NAME = "/tmp/predictions.db"
else:
    DB_NAME = os.path.join(BASE_DIR, "predictions.db")


def init_db():
    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        input_features TEXT,
        prediction TEXT,
        confidence REAL
    )
    """)

    conn.commit()
    conn.close()


def save_prediction(
    timestamp,
    input_features,
    prediction,
    confidence
):
    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
    INSERT INTO predictions
    (
        timestamp,
        input_features,
        prediction,
        confidence
    )
    VALUES (?, ?, ?, ?)
    """,
    (
        timestamp,
        input_features,
        prediction,
        confidence
    ))

    conn.commit()
    conn.close()