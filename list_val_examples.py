import csv

with open("data/splits.csv", newline="") as fh:
    rows = [r for r in csv.DictReader(fh) if r["split"] == "val"]
for label in ["negative", "recoverable_defects", "non_recoverable_defects"]:
    for r in [x for x in rows if x["label"] == label][:3]:
        print(label, r["path"])