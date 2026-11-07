
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

from preprocessing import load_gray, denoise_gaussian, enhance_clahe
from detection import threshold_otsu, clean_mask, find_regions, draw_regions


# Use the actual dataset folder from your project
DATA_DIR = Path("data/raw/carbon-multiclass")

CLASSES = [
    "negative",
    "recoverable_defects",
    "non_recoverable_defects"
]

# Stage 4: choose the threshold direction
DARK_DEFECTS = True

MIN_AREA = 30
IMAGE_INDEX = 0


def analyse(path):
    gray = load_gray(path)
    enhanced = enhance_clahe(denoise_gaussian(gray, 5))

    _, mask = threshold_otsu(
        enhanced,
        dark_defects=DARK_DEFECTS
    )

    cleaned = clean_mask(mask)
    regions = find_regions(cleaned, MIN_AREA)

    return enhanced, mask, cleaned, regions


# Part 1: display one image from each class
fig, axes = plt.subplots(3, 4, figsize=(13, 10))

for row, name in enumerate(CLASSES):
    files = sorted(
        list((DATA_DIR / name).glob("*.jpg"))
        + list((DATA_DIR / name).glob("*.jpeg"))
        + list((DATA_DIR / name).glob("*.png"))
    )

    if not files:
        raise SystemExit(
            f"No supported images found in {DATA_DIR / name}"
        )

    enhanced, mask, cleaned, regions = analyse(files[IMAGE_INDEX])

    total = sum(r["area"] for r in regions)

    print(
        f"{name:25s} regions={len(regions)} "
        f"total_area={total:.0f} px"
    )

    shown = [
        ("Enhanced", enhanced),
        ("Raw mask", mask),
        ("Cleaned mask", cleaned),
        (
            f"{len(regions)} regions",
            cv2.cvtColor(
                draw_regions(enhanced, regions),
                cv2.COLOR_BGR2RGB
            )
        )
    ]

    for col, (title, image) in enumerate(shown):
        if image.ndim == 2:
            axes[row, col].imshow(
                image, cmap="gray", vmin=0, vmax=255
            )
        else:
            axes[row, col].imshow(image)

        axes[row, col].set_title(
            f"{name}\n{title}" if col == 0 else title,
            fontsize=9
        )
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()


# Part 2: count images flagged by the baseline
print(
    "\nImages with at least 1 region, "
    "out of 50 (untuned baseline preview):"
)

for name in CLASSES:
    files = sorted(
        list((DATA_DIR / name).glob("*.jpg"))
        + list((DATA_DIR / name).glob("*.jpeg"))
        + list((DATA_DIR / name).glob("*.png"))
    )[:50]

    flagged = sum(
        1 for f in files if len(analyse(f)[3]) > 0
    )

    print(f"{name:25s} {flagged}/{len(files)}")
