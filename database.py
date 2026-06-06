import sqlite3

DB_NAME = "predictions.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        input_features TEXT,
        prediction INTEGER,
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