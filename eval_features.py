
import csv
import statistics
import cv2

from preprocessing import load_gray, denoise_gaussian, enhance_clahe
from detection import detect_edges, white_fraction, pipeline_score
from evaluation import (
    confusion_counts,
    precision_recall_f1,
    balanced_accuracy,
    auc,
    auc_interval,
)

TASKS = {
    "all defects vs good": (
        "negative",
        ["recoverable_defects", "non_recoverable_defects"],
    ),
    "non-recoverable vs good": (
        "negative",
        ["non_recoverable_defects"],
    ),
}

FEATURES = ["area_dark", "area_bright", "edge_density", "sharpness"]


def load_rows(split):
    with open("data/splits.csv", newline="", encoding="utf-8") as fh:
        return [
            r for r in csv.DictReader(fh)
            if r["split"] == split
        ]


def compute(rows):
    out = []

    for r in rows:
        gray = load_gray(r["path"])
        enhanced = enhance_clahe(denoise_gaussian(gray, 5))

        out.append({
            "label": r["label"],
            "edge_density": white_fraction(detect_edges(enhanced)),
            "sharpness": float(
                cv2.Laplacian(gray, cv2.CV_64F).var()
            ),
            "gray_mean": float(gray.mean()),
            "area_dark": pipeline_score(
                gray, dark_defects=True
            ),
            "area_bright": pipeline_score(
                gray, dark_defects=False
            ),
        })

    return out


def pick(data, task):
    good, bad = TASKS[task]

    chosen = [
        d for d in data
        if d["label"] == good or d["label"] in bad
    ]

    y = [
        0 if d["label"] == good else 1
        for d in chosen
    ]

    return chosen, y


def best_cutoff(scores, y):
    best_ba, best_t = -1.0, 0.0

    for t in sorted(set(scores)):
        y_pred = [1 if s > t else 0 for s in scores]

        tp, fp, fn, tn = confusion_counts(y, y_pred)
        ba = balanced_accuracy(tp, fp, fn, tn)

        if ba > best_ba:
            best_ba, best_t = ba, t

    return best_ba, best_t


# Load only train and validation data.
# The test split remains untouched.
train = compute(load_rows("train"))
val = compute(load_rows("val"))

print("Median brightness (gray_mean) per class on val:")

for name in [
    "negative",
    "recoverable_defects",
    "non_recoverable_defects",
]:
    vals = [
        d["gray_mean"]
        for d in val
        if d["label"] == name
    ]

    if vals:
        print(f"  {name:25s} {statistics.median(vals):7.1f}")
    else:
        print(f"  {name:25s} No images found")


for task in TASKS:
    tr, y_tr = pick(train, task)
    va, y_va = pick(val, task)

    n_bad = sum(y_va)

    print(
        f"\nTask: {task} "
        f"(val: {n_bad} defect / {len(y_va) - n_bad} good)"
    )

    # Baseline: predict every image as defective.
    tp, fp, fn, tn = confusion_counts(
        y_va, [1] * len(y_va)
    )

    reference_f1 = precision_recall_f1(tp, fp, fn)[2]
    reference_ba = balanced_accuracy(tp, fp, fn, tn)

    print(
        f"  'flag everything' reference: "
        f"F1={reference_f1:.3f} "
        f"balanced accuracy={reference_ba:.3f}"
    )

    for feat in FEATURES:
        s_tr = [d[feat] for d in tr]
        s_va = [d[feat] for d in va]

        # Select the cutoff using training data only.
        train_ba, cutoff = best_cutoff(s_tr, y_tr)

        # Evaluate the chosen cutoff on validation data.
        y_pred = [
            1 if s > cutoff else 0
            for s in s_va
        ]

        tp, fp, fn, tn = confusion_counts(y_va, y_pred)

        p, r, f1 = precision_recall_f1(tp, fp, fn)
        ba = balanced_accuracy(tp, fp, fn, tn)

        # Calculate validation AUC and its bootstrap range.
        pos = [
            s for s, y in zip(s_va, y_va)
            if y == 1
        ]

        neg = [
            s for s, y in zip(s_va, y_va)
            if y == 0
        ]

        lo, hi = auc_interval(pos, neg)

        print(
            f"  {feat:13s} cutoff={cutoff:9.3f}  "
            f"trainBA={train_ba:.3f}  "
            f"valP={p:.3f}  valR={r:.3f}  "
            f"valF1={f1:.3f}  valBA={ba:.3f}  "
            f"valAUC={auc(pos, neg):.3f} "
            f"(95% range {lo:.2f}-{hi:.2f})"
        )
