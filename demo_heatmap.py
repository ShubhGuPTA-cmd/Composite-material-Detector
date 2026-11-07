
from pathlib import Path

import matplotlib.pyplot as plt
import cv2

from preprocessing import (
    load_gray,
    denoise_gaussian,
    enhance_clahe,
)

from detection import (
    detect_edges,
    white_fraction,
    edge_density_map,
    hot_spots,
    draw_regions,
)

DATA_DIR = Path("data/raw/carbon-multiclass")

CLASSES = [
    "negative",
    "recoverable_defects",
    "non_recoverable_defects",
]

IMAGE_INDEX = 10
WINDOW = 31
K = 1.5

fig, axes = plt.subplots(3, 4, figsize=(14, 10))

for row, name in enumerate(CLASSES):
    files = sorted((DATA_DIR / name).glob("*.jpg"))

    if not files:
        raise SystemExit(
            f"No .jpg files in {DATA_DIR / name}"
        )

    if IMAGE_INDEX >= len(files):
        raise SystemExit(
            f"IMAGE_INDEX {IMAGE_INDEX} is too large "
            f"for class {name}, which has {len(files)} images"
        )

    gray = load_gray(files[IMAGE_INDEX])

    enhanced = enhance_clahe(
        denoise_gaussian(gray, 5)
    )

    edges = detect_edges(enhanced)
    density = edge_density_map(gray, WINDOW)

    cutoff, mask, regions = hot_spots(density, K)

    print(
        f"{name:25s} "
        f"image edge_density={white_fraction(edges):.3f}  "
        f"local max={density.max():.2f}  "
        f"hot-spot regions={len(regions)}"
    )

    overlay = cv2.cvtColor(
        draw_regions(gray, regions),
        cv2.COLOR_BGR2RGB
    )

    axes[row, 0].imshow(
        gray, cmap="gray", vmin=0, vmax=255
    )
    axes[row, 0].set_title(name, fontsize=9)

    axes[row, 1].imshow(edges, cmap="gray")
    axes[row, 1].set_title("Canny edges", fontsize=9)

    axes[row, 2].imshow(density, cmap="jet", vmin=0, vmax=0.3)
    axes[row, 2].set_title(
        "Local edge density (red = high)", fontsize=9
    )

    axes[row, 3].imshow(overlay)
    axes[row, 3].set_title(
        f"{len(regions)} hot spots", fontsize=9
    )

    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()
plt.show()
