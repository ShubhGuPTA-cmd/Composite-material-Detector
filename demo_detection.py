
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from preprocessing import load_gray, denoise_gaussian, enhance_clahe
from detection import (
    threshold_otsu,
    threshold_adaptive,
    detect_edges,
    white_fraction,
)

DATA_DIR = Path("data/raw/carbon-multiclass")

CLASSES = [
    "negative",
    "recoverable_defects",
    "non_recoverable_defects",
]

IMAGE_INDEX = 0


def prepare(path):
    gray = load_gray(path)
    blurred = denoise_gaussian(gray, 5)
    return enhance_clahe(blurred)


fig, axes = plt.subplots(3, 5, figsize=(15, 9))

print("Share of white pixels in each mask:")

for row, name in enumerate(CLASSES):
    folder = DATA_DIR / name

    files = sorted(
        list(folder.glob("*.jpg"))
        + list(folder.glob("*.jpeg"))
        + list(folder.glob("*.png"))
    )

    if not files:
        raise SystemExit(f"No images found in {folder}")

    img = prepare(files[IMAGE_INDEX])

    _, otsu_dark = threshold_otsu(img, dark_defects=True)
    _, otsu_bright = threshold_otsu(img, dark_defects=False)
    adaptive = threshold_adaptive(img)
    edges = detect_edges(img)

    print(
        f"{name:25s} "
        f"otsu_dark={white_fraction(otsu_dark):.3f} "
        f"otsu_bright={white_fraction(otsu_bright):.3f} "
        f"adaptive={white_fraction(adaptive):.3f} "
        f"edges={white_fraction(edges):.3f}"
    )

    panels = [
        (name, img),
        ("Otsu (dark)", otsu_dark),
        ("Otsu (bright)", otsu_bright),
        ("Adaptive (dark)", adaptive),
        ("Canny edges", edges),
    ]

    for col, (title, image) in enumerate(panels):
        axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row, col].set_title(title, fontsize=9)
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()


print("\nAverage share of white pixels over 50 images (Otsu, dark):")

for name in CLASSES:
    folder = DATA_DIR / name

    files = sorted(
        list(folder.glob("*.jpg"))
        + list(folder.glob("*.jpeg"))
        + list(folder.glob("*.png"))
    )[:50]

    if not files:
        raise SystemExit(f"No images found in {folder}")

    fractions = [
        white_fraction(threshold_otsu(prepare(f), True)[1])
        for f in files
    ]

    print(f"{name:25s} {sum(fractions) / len(fractions):.3f}")


def clean_mask(mask, open_size=3, close_size=7):
    """Remove tiny specks, then fill small gaps."""
    open_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_size, open_size)
    )
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_k)
    return cleaned


def find_regions(mask, min_area=30):
    """Find and measure white regions in the mask."""
    result = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = result[-2]
    regions = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)

        regions.append({
            "contour": c,
            "area": area,
            "bbox": (x, y, w, h)
        })

    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions


def draw_regions(image, regions):
    """Draw green contours and red bounding boxes."""
    if len(image.shape) == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    for r in regions:
        x, y, w, h = r["bbox"]

        cv2.drawContours(
            out, [r["contour"]], -1, (0, 255, 0), 1
        )

        cv2.rectangle(
            out, (x, y), (x + w, y + h), (0, 0, 255), 1
        )

    return out

