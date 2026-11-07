
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

DATA_DIR = Path("data/raw/carbon-multiclass")

CLASSES = [
    "negative",
    "recoverable_defects",
    "non_recoverable_defects",
]

# Check whether the dataset folder exists
if not DATA_DIR.exists():
    raise SystemExit(
        f"Folder not found: {DATA_DIR}. Run from the project root."
    )

# Part 1: Count images in each class
for name in CLASSES:
    files = sorted((DATA_DIR / name).glob("*.jpg"))
    print(name, "->", len(files), "images")

# Part 2: Read one image
first = sorted((DATA_DIR / "negative").glob("*.jpg"))[0]

img = cv2.imread(str(first))

if img is None:
    raise SystemExit(f"Could not read image: {first}")

print("File:", first.name)
print("Shape (height, width, channels):", img.shape)
print("Data type:", img.dtype)
print("Smallest / largest pixel value:", img.min(), img.max())
print("Pixel at row 100, col 100 (B, G, R):", img[100, 100])

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Grayscale shape:", gray.shape)

# Part 3: Display three images from each class
fig, axes = plt.subplots(3, 3, figsize=(9, 9))

for row, name in enumerate(CLASSES):
    files = sorted((DATA_DIR / name).glob("*.jpg"))[:3]

    for col, f in enumerate(files):
        image = cv2.imread(str(f))

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        axes[row, col].imshow(rgb)
        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_title(name)

plt.tight_layout()
plt.show()
