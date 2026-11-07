
import csv
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import load_gray
from detection import edge_density_map, hot_spots
from evaluation import iou_dice

ANNOT_DIR = Path("data/annotations")
WINDOW, K, MIN_AREA = 31, 1.5, 60
MARGIN = 15

with open("data/splits.csv", newline="") as fh:
    path_by_stem = {
        Path(r["path"]).stem: r["path"]
        for r in csv.DictReader(fh)
    }

mask_files = sorted(ANNOT_DIR.glob("*_mask.png"))

if not mask_files:
    raise SystemExit(
        "No annotation files found in data/annotations"
    )

fig, axes = plt.subplots(
    1, len(mask_files),
    figsize=(3.2 * len(mask_files), 3.8)
)
axes = np.atleast_1d(axes)
results = []

for ax, mf in zip(axes, mask_files):
    stem = mf.stem.removesuffix("_mask")

    if stem not in path_by_stem:
        raise SystemExit(
            f"Image {stem} not found in data/splits.csv"
        )

    gray = load_gray(path_by_stem[stem])

    truth = cv2.imread(
        str(mf), cv2.IMREAD_GRAYSCALE
    )

    if truth is None or truth.shape != gray.shape:
        raise SystemExit(
            f"Mask {mf.name} is missing or has the wrong size"
        )

    density = edge_density_map(gray, WINDOW)
    _, _, regions = hot_spots(density, K, MIN_AREA, MARGIN)

    pred = np.zeros(gray.shape, dtype=np.uint8)

    if regions:
        cv2.drawContours(
            pred,
            [r["contour"] for r in regions],
            -1,
            255,
            thickness=cv2.FILLED
        )

    iou, dice = iou_dice(pred, truth)

    t = float((truth > 0).mean())
    p = float((pred > 0).mean())

    denom = p + t - p * t
    chance_iou = p * t / denom if denom > 0 else 0.0

    results.append((iou, dice, chance_iou, t))

    print(
        f"{stem:9s} truth={t*100:5.1f}%  "
        f"pred={p*100:5.1f}%  IoU={iou:.3f}  "
        f"Dice={dice:.3f}  chanceIoU={chance_iou:.3f}  "
        f"wholeImageIoU={t:.3f}"
    )

    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    for m, color in [
        (truth, (255, 0, 0)),
        (pred, (0, 255, 0))
    ]:
        outlines = cv2.findContours(
            (m > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )[-2]

        cv2.drawContours(rgb, outlines, -1, color, 1)

    ax.imshow(rgb)
    ax.set_title(f"{stem}\nIoU {iou:.2f}", fontsize=9)
    ax.axis("off")

n = len(results)

print(
    f"\nMEAN over {n} images:  "
    f"IoU={sum(r[0] for r in results)/n:.3f}  "
    f"Dice={sum(r[1] for r in results)/n:.3f}  "
    f"chanceIoU={sum(r[2] for r in results)/n:.3f}  "
    f"wholeImageIoU={sum(r[3] for r in results)/n:.3f}"
)

Path("reports").mkdir(exist_ok=True)

plt.tight_layout()
plt.savefig(f"reports/localization_check_margin{MARGIN}.png", dpi=150)
print(f"Saved picture: reports/localization_check_margin{MARGIN}.png")
plt.show()
