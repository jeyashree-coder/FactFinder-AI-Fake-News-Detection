import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "factfinder.db"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection():
    ensure_data_dir()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_type TEXT NOT NULL,
                input_value TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_details TEXT,
                created_at DATETIME NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS bbc_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                published_date TEXT,
                source TEXT NOT NULL,
                collected_at DATETIME NOT NULL,
                url TEXT NOT NULL UNIQUE
            )
            """
        )


def save_prediction(input_type: str, input_value: str, prediction: str, confidence: float, model_details: str) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO predictions (input_type, input_value, prediction, confidence, model_details, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (input_type, input_value, prediction, confidence, model_details, datetime.utcnow().isoformat()),
        )


def get_prediction_history(limit: int = 50):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, input_type, input_value, prediction, confidence, model_details, created_at
            FROM predictions
            ORDER BY datetime(created_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def insert_bbc_article(title: str, content: str, published_date: str | None, source: str, collected_at: str, url: str) -> bool:
    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO bbc_news (title, content, published_date, source, collected_at, url)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (title, content, published_date, source, collected_at, url),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def get_today_bbc_news():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, title, content, published_date, source, collected_at, url
            FROM bbc_news
            ORDER BY collected_at DESC
            LIMIT 50
            """
        ).fetchall()
    return [dict(row) for row in rows]

def get_all_bbc_news():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT title, content
            FROM bbc_news
            """
        ).fetchall()
    return [dict(row) for row in rows]
