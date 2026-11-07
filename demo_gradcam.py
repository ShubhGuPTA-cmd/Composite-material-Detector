
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import CompositeDataset
from gradcam import GradCAM
from model import build_model
from evaluation import iou_dice


ANNOT_DIR = Path("data/annotations")
THRESHOLD = 0.5
TARGET_CLASS = 2

# Load the neutral model from Stage 20
model = build_model(pretrained=False)
model.load_state_dict(
    torch.load(
        "models/resnet18_neutral.pt",
        map_location="cpu",
        weights_only=True
    )
)
model.eval()

cam_tool = GradCAM(model)

# Use training images for this initial demonstration
dataset = CompositeDataset("train", neutral=True)

index_by_stem = {
    Path(r["path"]).stem: i
    for i, r in enumerate(dataset.rows)
}

mask_files = sorted(ANNOT_DIR.glob("*_mask.png"))

if not mask_files:
    raise SystemExit(
        "No annotation files found in data/annotations"
    )

fig, axes = plt.subplots(
    2, len(mask_files),
    figsize=(3.2 * len(mask_files), 6.6)
)
axes = np.array(axes).reshape(2, -1)

results = []


def outlines(mask):
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


for col, mf in enumerate(mask_files):
    stem = mf.stem.removesuffix("_mask")

    if stem not in index_by_stem:
        raise SystemExit(
            f"{stem} is not in the train split"
        )

    idx = index_by_stem[stem]
    tensor, _ = dataset[idx]

    with torch.enable_grad():
        cam, probs, _ = cam_tool(tensor, TARGET_CLASS)

    truth = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)

    if truth is None:
        raise SystemExit(f"Could not read mask: {mf}")

    if truth.shape != cam.shape:
        truth = cv2.resize(
            truth,
            (cam.shape[1], cam.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    pred = (cam >= THRESHOLD).astype(np.uint8) * 255

    iou, dice = iou_dice(pred, truth)

    t = float((truth > 0).mean())
    p = float((pred > 0).mean())

    denom = p + t - p * t
    chance = p * t / denom if denom > 0 else 0.0

    results.append((iou, dice, chance, t))

    print(
        f"{stem:9s} P(non-rec)={probs[2]:.2f}  "
        f"truth={t*100:5.1f}%  cam={p*100:5.1f}%  "
        f"IoU={iou:.3f}  Dice={dice:.3f}  "
        f"chanceIoU={chance:.3f}  wholeImageIoU={t:.3f}"
    )

    gray = cv2.resize(
        cv2.imread(dataset.rows[idx]["path"],
                   cv2.IMREAD_GRAYSCALE),
        (224, 224)
    )

    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    cv2.drawContours(
        rgb, outlines(truth), -1, (255, 0, 0), 1
    )
    cv2.drawContours(
        rgb, outlines(pred), -1, (0, 255, 0), 1
    )

    axes[0, col].imshow(rgb)
    axes[0, col].set_title(
        f"{stem} IoU {iou:.2f}", fontsize=9
    )

    axes[1, col].imshow(
        cam, cmap="jet", vmin=0, vmax=1
    )
    axes[1, col].set_title(
        f"Grad-CAM P(non-rec)={probs[2]:.2f}",
        fontsize=9
    )

    axes[0, col].axis("off")
    axes[1, col].axis("off")


n = len(results)

print(
    f"\nMEAN over {n} images: "
    f"IoU={sum(r[0] for r in results)/n:.3f}  "
    f"Dice={sum(r[1] for r in results)/n:.3f}  "
    f"chanceIoU={sum(r[2] for r in results)/n:.3f}  "
    f"wholeImageIoU={sum(r[3] for r in results)/n:.3f}"
)

print("Classic edge-density hot spots (Stage 16): IoU=0.131 Dice=0.206")

Path("reports").mkdir(exist_ok=True)

plt.tight_layout()
plt.savefig("reports/gradcam_check.png", dpi=150)

print("Saved picture: reports/gradcam_check.png")

plt.show()
