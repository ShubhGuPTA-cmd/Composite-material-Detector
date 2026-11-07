
from pathlib import Path
import matplotlib.pyplot as plt

from preprocessing import (
    load_gray,
    denoise_gaussian,
    denoise_median,
    denoise_bilateral,
    enhance_hist_eq,
    enhance_clahe,
)

DATA_DIR = Path("data/raw/carbon-multiclass/negative")

IMAGE_INDEX = 10

files = sorted(DATA_DIR.glob("*.jpg"))

if not files:
    raise SystemExit(
        f"No .jpg files found in {DATA_DIR}"
    )

if IMAGE_INDEX >= len(files):
    raise SystemExit(
        f"IMAGE_INDEX is {IMAGE_INDEX}, but only "
        f"{len(files)} images were found."
    )

gray = load_gray(files[IMAGE_INDEX])

print("Using image:", files[IMAGE_INDEX].name)

gaussian = denoise_gaussian(gray, 11)
median = denoise_median(gray, 5)
bilateral = denoise_bilateral(gray)

hist_eq = enhance_hist_eq(gray)
clahe = enhance_clahe(gray)
gauss_clahe = enhance_clahe(gaussian)

print("Contrast (standard deviation of pixels):")
print("  Original :", round(float(gray.std()), 1))
print("  Hist. eq.:", round(float(hist_eq.std()), 1))
print("  CLAHE    :", round(float(clahe.std()), 1))

panels = [
    ("Original", gray),
    ("Gaussian 5x5", gaussian),
    ("Median 5x5", median),
    ("Bilateral", bilateral),
    ("Hist. equalization", hist_eq),
    ("CLAHE", clahe),
    ("Gaussian + CLAHE", gauss_clahe),
]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for ax, (title, image) in zip(axes.ravel(), panels):
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")

hist_ax = axes[1, 3]
hist_ax.hist(gray.ravel(), bins=64)
hist_ax.set_title("Brightness histogram (original)")
hist_ax.set_xlabel("Pixel value (0-255)")
hist_ax.set_ylabel("Number of pixels")

plt.tight_layout()
plt.show()
