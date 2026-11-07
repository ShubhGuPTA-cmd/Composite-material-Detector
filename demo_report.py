import csv
from pathlib import Path
from pipeline import Inspector
from report import make_report
from severity import DEFAULT_RULES

inspector = Inspector()
with open("data/splits.csv", newline="") as fh:
    val_rows = [r for r in csv.DictReader(fh) if r["split"] == "val"]
Path("reports").mkdir(exist_ok=True)
for label in ["negative", "recoverable_defects", "non_recoverable_defects"]:
    row = next(r for r in val_rows if r["label"] == label)
    result = inspector.inspect(row["path"])
    pdf = make_report(result, Path(row["path"]).name, DEFAULT_RULES)
    out = Path("reports") / f"example_{label}.pdf"
    out.write_bytes(pdf)
    print(f"{label:24s} -> {out}  ({len(pdf) / 1024:.0f} KB)  "
          f"severity={result['verdict']['severity']}")