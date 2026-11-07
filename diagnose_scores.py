
import csv
import statistics
import cv2

from preprocessing import load_gray, denoise_gaussian, enhance_clahe
from detection import pipeline_score, baseline_detect, detect_edges, white_fraction
from evaluation import auc


with open("data/splits.csv", newline="", encoding="utf-8") as fh:
    rows = [
        r for r in csv.DictReader(fh)
        if r["split"] == "train"
    ]

print("train images:", len(rows))


def features(gray):
    enhanced = enhance_clahe(denoise_gaussian(gray, 5))

    return {
        "area_dark": pipeline_score(gray, dark_defects=True),
        "area_bright": pipeline_score(gray, dark_defects=False),
        "regions_dark": float(
            len(baseline_detect(gray, dark_defects=True))
        ),
        "edge_density": white_fraction(detect_edges(enhanced)),
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "sharpness": float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        ),
    }


groups = {
    "negative": [],
    "recoverable_defects": [],
    "non_recoverable_defects": [],
}

for row in rows:
    label = row["label"]

    if label not in groups:
        raise ValueError(f"Unexpected label: {label}")

    groups[label].append(features(load_gray(row["path"])))


if any(not values for values in groups.values()):
    raise ValueError("At least one class has no training images.")

names = list(groups["negative"][0].keys())

print(
    f"\n{'feature':14s}{'med neg':>10s}"
    f"{'med recov':>11s}{'med nonrec':>12s}"
    f"{'AUC def':>9s}{'AUC nonrec':>12s}"
)
print("-" * 68)

for name in names:
    neg = [f[name] for f in groups["negative"]]
    rec = [f[name] for f in groups["recoverable_defects"]]
    non = [f[name] for f in groups["non_recoverable_defects"]]

    auc_defect = auc(rec + non, neg)
    auc_nonrec = auc(non, neg)

    print(
        f"{name:14s}"
        f"{statistics.median(neg):10.3f}"
        f"{statistics.median(rec):11.3f}"
        f"{statistics.median(non):12.3f}"
        f"{auc_defect:9.3f}"
        f"{auc_nonrec:12.3f}"
    )
