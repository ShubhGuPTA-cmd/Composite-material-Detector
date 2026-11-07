
import csv
import time

from evaluation import confusion_counts, precision_recall_f1
from preprocessing import load_gray
from detection import baseline_detect

SPLIT = "val"

with open("data/splits.csv", newline="", encoding="utf-8") as fh:
    rows = [
        r for r in csv.DictReader(fh)
        if r["split"] == SPLIT
    ]

print(f"Images in '{SPLIT}':", len(rows))

# Load every validation image once.
images = [
    (load_gray(r["path"]), r["label"])
    for r in rows
]

y_true = [
    0 if label == "negative" else 1
    for _, label in images
]

# Reference: a system that flags every image.
tp, fp, fn, tn = confusion_counts(
    y_true, [1] * len(y_true)
)
p, r, f = precision_recall_f1(tp, fp, fn)

print(
    f"\nReference 'flag everything': "
    f"precision={p:.3f} recall={r:.3f} F1={f:.3f}\n"
)

# Test both threshold directions.
for dark in [True, False]:
    start = time.perf_counter()

    y_pred = [
        1 if len(baseline_detect(g, dark_defects=dark)) > 0 else 0
        for g, _ in images
    ]

    ms = (
        (time.perf_counter() - start) / len(images) * 1000
        if images else 0.0
    )

    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    p, r, f = precision_recall_f1(tp, fp, fn)

    print(f"DARK_DEFECTS={dark}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(
        f"  precision={p:.3f} recall={r:.3f} "
        f"F1={f:.3f} ({ms:.1f} ms/image)"
    )

    for name in [
        "negative",
        "recoverable_defects",
        "non_recoverable_defects",
    ]:
        flags = [
            pred
            for (_, label), pred in zip(images, y_pred)
            if label == name
        ]

        print(
            f"  flagged in {name:25s} "
            f"{sum(flags)}/{len(flags)}"
        )

    print()
