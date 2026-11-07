
import statistics
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from preprocessing import load_gray
from dataset import CompositeDataset
from model import build_model
from gradcam import GradCAM
from occlusion import occlusion_map
from detection import edge_density_map, hot_spots
from evaluation import iou_dice


ANNOT_DIR = Path("data/annotations_val")
TARGET_CLASS = 2
THRESHOLD = 0.5

WINDOW, K, MIN_AREA, MARGIN = 31, 1.5, 60, 15
METHODS = ["classic", "gradcam", "occlusion"]


def load_model():
    model = build_model(pretrained=False)
    model.load_state_dict(
        torch.load("models/resnet18_neutral.pt", map_location="cpu")
    )
    return model.eval()


def outlines(mask):
    return cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )[-2]


cam_tool = GradCAM(load_model())
occ_model = load_model()
dataset = CompositeDataset("val", neutral=True)

index_by_stem = {
    Path(row["path"]).stem: i
    for i, row in enumerate(dataset.rows)
}

mask_files = sorted(ANNOT_DIR.glob("*_mask.png"))

if not mask_files:
    raise SystemExit("No annotation files in data/annotations_val")

n = len(mask_files)
fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 7.2))
axes = np.array(axes).reshape(3, -1)

results = []

for col, mf in enumerate(mask_files):
    stem = mf.stem.removesuffix("_mask")

    if stem not in index_by_stem:
        raise SystemExit(f"{stem} is not in the val split")

    idx = index_by_stem[stem]
    tensor, _ = dataset[idx]
    gray = load_gray(dataset.rows[idx]["path"])

    truth = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)

    if truth is None or truth.shape != gray.shape:
        raise SystemExit(
            f"Mask {mf.name} is missing or has the wrong size"
        )

    # Method 1: Classic edge-density hot spots
    density = edge_density_map(gray, WINDOW)
    _, _, regions = hot_spots(density, K, MIN_AREA, MARGIN)

    classic = np.zeros(gray.shape, dtype=np.uint8)

    if regions:
        cv2.drawContours(
            classic,
            [r["contour"] for r in regions],
            -1,
            255,
            thickness=cv2.FILLED
        )

    # Method 2: Grad-CAM
    cam, _, _ = cam_tool(tensor, TARGET_CLASS)

    # Method 3: Occlusion sensitivity
    occ, _ = occlusion_map(occ_model, tensor, TARGET_CLASS)

    preds = {
        "classic": classic,
        "gradcam": (cam >= THRESHOLD).astype(np.uint8) * 255,
        "occlusion": (occ >= THRESHOLD).astype(np.uint8) * 255
    }

    t = float((truth > 0).mean())
    entry = {"stem": stem, "truth": t}

    line = f"{stem:8s} truth={t*100:4.1f}% whole={t:.3f}"

    for row, name in enumerate(METHODS):
        iou, dice = iou_dice(preds[name], truth)
        p = float((preds[name] > 0).mean())

        denom = p + t - p * t

        entry[name] = (
            iou,
            dice,
            p,
            p * t / denom if denom > 0 else 0.0
        )

        line += f" | {name} IoU={iou:.3f} ({p*100:4.1f}%)"

        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Red = manually drawn annotation
        cv2.drawContours(
            rgb, outlines(truth), -1, (255, 0, 0), 1
        )

        # Green = method prediction
        cv2.drawContours(
            rgb, outlines(preds[name]), -1, (0, 255, 0), 1
        )

        axes[row, col].imshow(rgb)
        axes[row, col].set_title(
            f"{stem}\n{name} {iou:.2f}",
            fontsize=7
        )
        axes[row, col].axis("off")

    print(line)
    results.append(entry)


print(f"\nMEAN over {n} validation images")
print(
    "  mark-everything reference: "
    f"IoU={statistics.mean(r['truth'] for r in results):.3f}"
)

for name in METHODS:
    mean_iou = statistics.mean(r[name][0] for r in results)
    mean_dice = statistics.mean(r[name][1] for r in results)
    mean_area = statistics.mean(r[name][2] for r in results)
    mean_chance = statistics.mean(r[name][3] for r in results)

    wins = sum(
        1 for r in results
        if r[name][0] > r["truth"]
    )

    print(
        f"  {name:10s} IoU={mean_iou:.3f} "
        f"Dice={mean_dice:.3f} "
        f"chanceIoU={mean_chance:.3f} "
        f"marked area={mean_area*100:.1f}% "
        f"beats mark-everything on {wins}/{n}"
    )

plt.tight_layout()
plt.savefig("reports/localization_val.png", dpi=150)
print("Saved picture: reports/localization_val.png")
plt.show()
