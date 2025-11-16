import os
from pathlib import Path

import pandas as pd
import pytest


def test_db_save_and_query(tmp_path, monkeypatch):
    # Use temp DB and raw dir
    db_path = tmp_path / "analytics.db"
    raw_dir = tmp_path / "raw"
    monkeypatch.setenv("GESTURE_DB_PATH", str(db_path))
    monkeypatch.setenv("GESTURE_RAW_DIR", str(raw_dir))

    # Import after setting env so module picks up overrides
    import db  # noqa: WPS433

    db.init_db()

    # Minimal dataframe
    df = pd.DataFrame(
        {
            "frame": [0, 0],
            "hand_index": [0, 0],
            "handedness": ["Unknown", "Unknown"],
            "landmark_id": [0, 1],
            "x": [0.5, 0.6],
            "y": [0.5, 0.6],
            "z": [0.0, 0.0],
        }
    )

    summary = {"frames_total": 1, "frames_with_hand": 1}
    row_id = db.save_analysis("sample.mp4", df, summary)
    assert row_id > 0

    catalog = db.query_catalog()
    assert len(catalog) == 1
    assert Path(catalog.iloc[0]["raw_data_csv"]).exists()
