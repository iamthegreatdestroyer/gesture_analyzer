import os
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Allow override for tests via environment variables
DB_PATH = Path(
    os.environ.get("GESTURE_DB_PATH", "data/analytics.db")
).resolve()
RAW_DIR = Path(os.environ.get("GESTURE_RAW_DIR", "data/raw")).resolve()


def _ensure_dirs() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    """Create the SQLite database and table if missing."""
    _ensure_dirs()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT NOT NULL,
                analysis_date TEXT NOT NULL,
                patterns_json TEXT NOT NULL,
                raw_data_csv TEXT NOT NULL,
                decoded_transcript TEXT
            )
            """
        )
        # Schema upgrade: ensure decoded_transcript column exists
        cur = conn.execute("PRAGMA table_info(video_analytics)")
        cols = [r[1] for r in cur.fetchall()]
        if "decoded_transcript" not in cols:
            conn.execute(
                "ALTER TABLE video_analytics ADD COLUMN decoded_transcript TEXT"
            )
        conn.commit()


def save_analysis(video_path: str, df: pd.DataFrame, summary: Dict) -> int:
    """
    Save analysis results: CSV for raw landmarks and JSON summary into the DB.

    Returns the inserted row id.
    """
    _ensure_dirs()
    video_name = Path(video_path).name
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # Save raw dataframe to CSV
    csv_name = f"{Path(video_name).stem}_{timestamp}.csv"
    csv_path = RAW_DIR / csv_name
    df.to_csv(csv_path, index=False)

    # Insert metadata into SQLite
    patterns_json = json.dumps(summary, ensure_ascii=False)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        decoded_transcript = summary.get("transcript_decoded", "")
        cur.execute(
            """
            INSERT INTO video_analytics (
                video_name,
                analysis_date,
                patterns_json,
                raw_data_csv,
                decoded_transcript
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                video_name,
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                patterns_json,
                str(csv_path),
                decoded_transcript,
            ),
        )
        conn.commit()
        row_id = cur.lastrowid or 0
        return int(row_id)


def query_catalog(limit: Optional[int] = None) -> pd.DataFrame:
    """Return the catalog as a pandas DataFrame."""
    _ensure_dirs()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM video_analytics ORDER BY id DESC",
            conn,
        )
        if limit is not None:
            return df.head(limit)
        return df
