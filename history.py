import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_FILE = Path("data/inspections.db")

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspected_at TEXT NOT NULL,
    file_name TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    severity TEXT NOT NULL,
    needs_review INTEGER NOT NULL,
    p_good REAL,
    p_recoverable REAL,
    p_nonrecoverable REAL,
    localized INTEGER NOT NULL,
    attention_percent REAL,
    region_count INTEGER NOT NULL,
    method TEXT,
    rules_json TEXT
)
"""


def _connect(db_file):
    Path(db_file).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row          # lets us read columns by name
    conn.execute(CREATE_TABLE)
    return conn


def save_inspection(result, file_name, rules, db_file=DB_FILE):
    """Store one inspection result (no images). Returns the new row number."""
    probs = result["class_probs"]
    when = result.get("inspected_at") or datetime.now()
    conn = _connect(db_file)
    try:
        with conn:                          # saves (commits) when the block ends
            cursor = conn.execute(
                "INSERT INTO inspections (inspected_at, file_name, predicted_class, severity,"
                " needs_review, p_good, p_recoverable, p_nonrecoverable, localized,"
                " attention_percent, region_count, method, rules_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (when.strftime("%Y-%m-%d %H:%M:%S"), file_name, result["predicted_class"],
                 result["verdict"]["severity"], int(result["verdict"]["needs_review"]),
                 probs["negative"], probs["recoverable_defects"],
                 probs["non_recoverable_defects"], int(result["localized"]),
                 result["cam_area_percent"], len(result["measurements"]),
                 result["method"], json.dumps(rules, sort_keys=True)))
        return cursor.lastrowid
    finally:
        conn.close()


def list_inspections(limit=50, db_file=DB_FILE):
    """Newest first."""
    conn = _connect(db_file)
    try:
        rows = conn.execute("SELECT * FROM inspections ORDER BY id DESC LIMIT ?",
                            (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_all(db_file=DB_FILE):
    conn = _connect(db_file)
    try:
        with conn:
            conn.execute("DELETE FROM inspections")
    finally:
        conn.close()