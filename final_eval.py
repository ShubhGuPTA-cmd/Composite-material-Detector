import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import CLASS_NAMES, make_loader
from detection import detect_edges, pipeline_score, white_fraction
from evaluation import (
    auc,
    auc_interval,
    balanced_accuracy,
    best_cutoff_by_ba,
    confusion_counts,
    precision_recall_f1,
)
from model import build_model, predict_probs
from pipeline import Inspector
from preprocessing import denoise_gaussian, enhance_clahe, load_gray
from severity import assess

LOCK = Path("reports/test_run.lock")
RESULTS = Path("reports/test_results.txt")

FEATURES = ["area_dark", "area_bright", "edge_density", "sharpness"]

MODELS = {
    "CNN normal": ("models/resnet18_best.pt", False),
    "CNN neutral": ("models/resnet18_neutral.pt", True),
}

TASKS = [
    "Task 1: all defects vs good",
    "Task 2: non-recoverable vs good",
]

if LOCK.exists() and "--force" not in sys.argv:
    raise SystemExit(
        f"The test split was already used on {LOCK.read_text().strip()}. "
        "Changing anything after seeing test results invalidates them. "
        "Use --force only for debugging, and say so in the report."
    )

lines = []


def out(text=""):
    print(text)
    lines.append(text)


def read_rows(split):
    with open("data/splits.csv", newline="") as fh:
        return [
            r for r in csv.DictReader(fh)
            if r["split"] == split
        ]


def label_array(rows):
    return np.array([CLASS_NAMES.index(r["label"]) for r in rows])


def views(labels):
    """Per task: which images take part, and which are defects."""
    return {
        TASKS[0]: (
            np.ones(len(labels), dtype=bool),
            (labels != 0).astype(int),
        ),
        TASKS[1]: (
            (labels == 0) | (labels == 2),
            (labels == 2).astype(int),
        ),
    }


def classic_features(path):
    gray = load_gray(path)
    enhanced = enhance_clahe(denoise_gaussian(gray, 5))

    return {
        "area_dark": pipeline_score(gray, dark_defects=True),
        "area_bright": pipeline_score(gray, dark_defects=False),
        "edge_density": white_fraction(detect_edges(enhanced)),
        "sharpness": float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        ),
    }


def metrics_line(name, scores, y_true, y_pred):
    pos = [s for s, y in zip(scores, y_true) if y == 1]
    neg = [s for s, y in zip(scores, y_true) if y == 0]

    a = auc(pos, neg)
    lo, hi = auc_interval(pos, neg)

    tp, fp, fn, tn = confusion_counts(
        list(y_true), list(y_pred)
    )
    p, r, _ = precision_recall_f1(tp, fp, fn)

    return (
        f"  {name:14s} AUC={a:.3f} "
        f"(95% range {lo:.2f}-{hi:.2f})  "
        f"balanced accuracy="
        f"{balanced_accuracy(tp, fp, fn, tn):.3f}  "
        f"precision={p:.3f}  recall={r:.3f}"
    )


train_rows = read_rows("train")
test_rows = read_rows("test")

train_labels = label_array(train_rows)
test_labels = label_array(test_rows)

train_views = views(train_labels)
test_views = views(test_labels)

print("Computing classic features (a minute or two)...")

train_feats = [
    classic_features(r["path"]) for r in train_rows
]
test_feats = [
    classic_features(r["path"]) for r in test_rows
]

cnn = {}

for name, (path, neutral) in MODELS.items():
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path))

    probs, labels = predict_probs(
        model,
        make_loader(
            "test", 32, allow_test=True, neutral=neutral
        ),
    )

    if not (labels.numpy() == test_labels).all():
        raise SystemExit(
            "The test image order does not match. Something is wrong."
        )

    cnn[name] = probs.numpy()

out(
    f"FINAL TEST EVALUATION   "
    f"{datetime.now():%Y-%m-%d %H:%M}   "
    f"test images: {len(test_rows)}"
)

for name, (path, _) in MODELS.items():
    fingerprint = hashlib.md5(
        Path(path).read_bytes()
    ).hexdigest()[:12]

    out(f"  {name}: {path}  md5 {fingerprint}")

for task in TASKS:
    mask_tr, y_tr = train_views[task]
    mask_te, y_te = test_views[task]
    yt = y_te[mask_te]

    out(
        f"\n{task}   "
        f"(test: {int(yt.sum())} defect / "
        f"{int((1 - yt).sum())} good; "
        "'flag everything' balanced accuracy = 0.500)"
    )

    for feat in FEATURES:
        s_tr = np.array(
            [f[feat] for f in train_feats]
        )[mask_tr]

        s_te = np.array(
            [f[feat] for f in test_feats]
        )[mask_te]

        cutoff = best_cutoff_by_ba(
            s_tr.tolist(), y_tr[mask_tr].tolist()
        )

        out(
            metrics_line(
                feat, s_te, yt, (s_te > cutoff).astype(int)
            )
        )

    for name, probs in cnn.items():
        guess = probs.argmax(axis=1)

        if task == TASKS[0]:
            scores = 1 - probs[:, 0]
            pred = (guess != 0).astype(int)
        else:
            scores = probs[:, 2]
            pred = (guess == 2).astype(int)

        out(
            metrics_line(
                name, scores[mask_te], yt, pred[mask_te]
            )
        )

for name, probs in cnn.items():
    guess = probs.argmax(axis=1)

    out(
        f"\n{name}: 3-class accuracy = "
        f"{(guess == test_labels).mean():.3f}"
    )

    out(
        "  confusion (rows = true class; columns = "
        "predicted: good / recoverable / non-recoverable)"
    )

    for i, cname in enumerate(CLASS_NAMES):
        counts = [
            int(((test_labels == i) & (guess == j)).sum())
            for j in range(3)
        ]
        out(f"  {cname:24s}{counts}")

out("\nSeverity with the default rules (CNN neutral):")

levels = ["None", "Low", "Medium", "High"]
expected = {0: "None", 1: "Low", 2: "High"}

table = {}
review_total = 0
wrong = 0
wrong_flagged = 0

for p, y in zip(cnn["CNN neutral"], test_labels):
    v = assess([float(x) for x in p])

    table[(y, v["severity"])] = (
        table.get((y, v["severity"]), 0) + 1
    )

    review_total += int(v["needs_review"])

    if v["severity"] != expected[y]:
        wrong += 1
        wrong_flagged += int(v["needs_review"])

out(
    f"  {'true class':26s}"
    + "".join(f"{lvl:>8s}" for lvl in levels)
)

for i, cname in enumerate(CLASS_NAMES):
    out(
        f"  {cname:26s}"
        + "".join(
            f"{table.get((i, lvl), 0):8d}"
            for lvl in levels
        )
    )

out(
    f"  needs-review flags: {review_total} of "
    f"{len(test_labels)} images; {wrong_flagged} of "
    f"{wrong} severity decisions that differ from the "
    "dataset class were flagged"
)

out("\nInspection time (full pipeline, CPU):")

inspector = Inspector()
inspector.inspect(test_rows[0]["path"])  # warm-up; not counted

times = [
    inspector.inspect(r["path"])["seconds"] * 1000
    for r in test_rows
]

out(
    f"  {len(times)} images: "
    f"median {np.median(times):.0f} ms, "
    f"mean {np.mean(times):.0f} ms, "
    f"95th percentile {np.percentile(times, 95):.0f} ms"
)

out(
    "\nSaved: reports/test_results.txt   "
    "(the test split is now marked as used)"
)

RESULTS.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

LOCK.write_text(
    f"{datetime.now():%Y-%m-%d %H:%M}"
)