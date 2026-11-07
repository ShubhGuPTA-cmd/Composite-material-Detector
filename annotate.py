
import csv
import sys
from pathlib import Path
import cv2
import numpy as np

# Usage:  python src\annotate.py          -> 5 train images (as before)
#         python src\annotate.py val 10   -> 10 validation images
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"
N_IMAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SCALE = 3
OUT_DIR = Path("data/annotations" if SPLIT == "train" else f"data/annotations_{SPLIT}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open("data/splits.csv", newline="") as fh:
    rows = [r for r in csv.DictReader(fh)
            if r["split"] == SPLIT and r["label"] == "non_recoverable_defects"]
rows = rows[:N_IMAGES]
if not rows:
    raise SystemExit(f"No {SPLIT} images found in data/splits.csv")


for r in rows:
    path = Path(r["path"])
    out_file = OUT_DIR / (path.stem + "_mask.png")

    if out_file.exists():
        print("Already done:", path.name)
        

    img = cv2.imread(str(path))

    if img is None:
        raise SystemExit(f"Could not read image: {path}")

    big = cv2.resize(
        img,
        None,
        fx=SCALE,
        fy=SCALE,
        interpolation=cv2.INTER_CUBIC
    )

    print(f"\nImage: {path.name}")
    print("Drag a box around each visible defect.")
    print("Press ENTER to confirm a box.")
    print("Press ESC when finished with this image.")

    boxes = cv2.selectROIs(
        "Draw defect boxes",
        big,
        showCrosshair=False,
        fromCenter=False
    )

    # Black background; white rectangles indicate defects
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for x, y, w, h in boxes:
        x1 = max(0, int(x / SCALE))
        y1 = max(0, int(y / SCALE))
        x2 = min(img.shape[1], int((x + w) / SCALE))
        y2 = min(img.shape[0], int((y + h) / SCALE))

        mask[y1:y2, x1:x2] = 255

    success = cv2.imwrite(str(out_file), mask)

    if not success:
        raise SystemExit(f"Could not save mask: {out_file}")

    print(f"Saved: {out_file}")
    print(f"Number of boxes: {len(boxes)}")

cv2.destroyAllWindows()
print("\nAnnotation script finished.")
