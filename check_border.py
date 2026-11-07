
import csv
import statistics
import cv2
import numpy as np

from preprocessing import load_gray
from detection import edge_density_map, hot_spots

CLASSES = [
    "negative",
    "recoverable_defects",
    "non_recoverable_defects"
]

MARGIN = 12
N_PER_CLASS = 30
WINDOW, K, MIN_AREA = 31, 1.5, 60

with open("data/splits.csv", newline="") as fh:
    rows = [
        r for r in csv.DictReader(fh)
        if r["split"] == "train"
    ]


def border_share(path):
    """Measure the share of predicted hot-spot pixels near image borders."""
    gray = load_gray(path)

    density = edge_density_map(gray, WINDOW)
    _, _, regions = hot_spots(density, K, MIN_AREA)

    pred = np.zeros(gray.shape, dtype=np.uint8)

    if regions:
        cv2.drawContours(
            pred,
            [r["contour"] for r in regions],
            -1,
            255,
            thickness=cv2.FILLED
        )

    hot = pred > 0

    border = np.ones(gray.shape, dtype=bool)
    border[MARGIN:-MARGIN, MARGIN:-MARGIN] = False

    if hot.sum() == 0:
        return None

    return (
        float(hot[border].sum() / hot.sum()),
        float(border.mean())
    )


for name in CLASSES:
    paths = [
        r["path"] for r in rows
        if r["label"] == name
    ][:N_PER_CLASS]

    results = [
        result
        for result in (border_share(p) for p in paths)
        if result is not None
    ]

    if not results:
        print(f"{name:25s} no hot spots found")
        continue

    measured = statistics.mean(
        r[0] for r in results
    ) * 100

    expected = statistics.mean(
        r[1] for r in results
    ) * 100

    print(
        f"{name:25s} images={len(results):3d}  "
        f"hot-spot pixels in border: {measured:5.1f}%   "
        f"expected if evenly spread: {expected:5.1f}%"
    )
