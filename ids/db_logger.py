# db_logger.py
from __future__ import annotations
from pathlib import Path
import os, json, sqlite3
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
_env = os.environ.get("IDS_DB_PATH", "").strip()
DB_PATH = Path(_env).resolve() if _env else (PROJECT_ROOT / "detections.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS logs(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TEXT,
      features TEXT,
      prediction TEXT,
      confidence TEXT
    );
    """)
    conn.commit()
    conn.close()
    print(f" DB ready at: {DB_PATH}")

def insert_log(features, prediction: str, confidence_list):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute(
        "INSERT INTO logs(ts, features, prediction, confidence) VALUES(?,?,?,?)",
        (
            datetime.utcnow().isoformat(timespec="seconds"),
            json.dumps(features, ensure_ascii=False),
            prediction,
            json.dumps(confidence_list),
        ),
    )
    conn.commit()
    conn.close()
