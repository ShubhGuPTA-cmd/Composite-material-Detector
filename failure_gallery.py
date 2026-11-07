
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline import Inspector
from visuals import attention_overlay

NICE = {
    "negative": "good",
    "recoverable_defects": "recoverable",
    "non_recoverable_defects": "non-recoverable",
}
N_SHOWN = 8
SEED = 0

with open("data/splits.csv", newline="") as fh:
    rows = [r for r in csv.DictReader(fh) if r["split"] == "test"]

inspector = Inspector()
print(f"Inspecting {len(rows)} test images (about a minute)...")

items = []
for row in rows:
    result = inspector.inspect(row["path"])
    items.append({
        "row": row,
        "result": result,
        "confidence": result["class_probs"][result["predicted_class"]],
    })

errors = [
    it for it in items
    if it["result"]["predicted_class"] != it["row"]["label"]
]
correct = [
    it for it in items
    if it["result"]["predicted_class"] == it["row"]["label"]
]

errors.sort(key=lambda it: it["confidence"], reverse=True)

rng = np.random.default_rng(SEED)
picked = [
    correct[i]
    for i in rng.choice(
        len(correct),
        size=min(N_SHOWN, len(correct)),
        replace=False,
    )
]

print(f"\nErrors: {len(errors)} of {len(items)}")

pairs = Counter(
    (
        NICE[it["row"]["label"]],
        NICE[it["result"]["predicted_class"]],
    )
    for it in errors
)

for (true, pred), n in pairs.most_common():
    print(f"  true {true:16s} -> predicted {pred:16s}: {n}")

print("\nAll errors, most confident first:")
for it in errors:
    print(
        f"  {Path(it['row']['path']).stem:10s} "
        f"true={NICE[it['row']['label']]:16s} "
        f"predicted={NICE[it['result']['predicted_class']]:16s} "
        f"confidence={it['confidence']:.0%}"
    )


def draw(shown, title, filename):
    cols = 4
    rows_n = max(1, (len(shown) + cols - 1) // cols)

    fig, axes = plt.subplots(
        rows_n, cols,
        figsize=(3.3 * cols, 3.9 * rows_n),
    )
    axes = np.array(axes).reshape(rows_n, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, it in zip(axes.ravel(), shown):
        res, row = it["result"], it["row"]
        ax.imshow(attention_overlay(res))
        ax.set_title(
            f"{Path(row['path']).stem}\n"
            f"true: {NICE[row['label']]}   "
            f"predicted: {NICE[res['predicted_class']]} "
            f"({it['confidence']:.0%})\n"
            f"severity: {res['verdict']['severity']}",
            fontsize=7,
        )

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)


draw(
    errors[:N_SHOWN],
    "Most confident mistakes (test set)",
    "reports/gallery_errors.png",
)
draw(
    picked,
    "Random correct predictions (test set, seed 0)",
    "reports/gallery_correct.png",
)

print("\nSaved: reports/gallery_errors.png and reports/gallery_correct.png")
