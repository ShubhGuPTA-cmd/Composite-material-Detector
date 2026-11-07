import json
import tempfile
from datetime import datetime
from pathlib import Path
from history import delete_all, list_inspections, save_inspection

def fake_result(severity, p_good, when):
    rest = (1 - p_good) / 2
    return {"predicted_class": "negative", "verdict": {"severity": severity, "needs_review": False},
            "class_probs": {"negative": p_good, "recoverable_defects": rest,
                            "non_recoverable_defects": rest},
            "localized": False, "cam_area_percent": 0.0, "measurements": [],
            "method": "test", "inspected_at": when}

rules = {"defect_threshold": 0.5}
hostile = "x'); DROP TABLE inspections;--.jpg"
checks = []
with tempfile.TemporaryDirectory() as tmp:
    db = Path(tmp) / "test.db"
    checks.append(("empty database lists nothing", list_inspections(db_file=db) == []))
    first = save_inspection(fake_result("None", 0.9, datetime(2026, 10, 21, 18, 0, 0)),
                            "a.jpg", rules, db_file=db)
    rows = list_inspections(db_file=db)
    checks.append(("first save gets row number 1", first == 1))
    checks.append(("saved values come back correctly",
                   rows[0]["file_name"] == "a.jpg" and rows[0]["severity"] == "None"
                   and abs(rows[0]["p_good"] - 0.9) < 1e-9
                   and rows[0]["inspected_at"] == "2026-10-21 18:00:00"))
    checks.append(("rules are stored as JSON", json.loads(rows[0]["rules_json"]) == rules))
    save_inspection(fake_result("High", 0.1, datetime(2026, 10, 21, 18, 5, 0)),
                    hostile, rules, db_file=db)
    rows = list_inspections(db_file=db)
    checks.append(("hostile file name is stored as plain text, table survives",
                   len(rows) == 2 and rows[0]["file_name"] == hostile))
    checks.append(("newest result comes first", rows[0]["severity"] == "High"))
    checks.append(("limit works", len(list_inspections(limit=1, db_file=db)) == 1))
    delete_all(db_file=db)
    checks.append(("delete_all empties the history", list_inspections(db_file=db) == []))

print("History self-check:")
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")