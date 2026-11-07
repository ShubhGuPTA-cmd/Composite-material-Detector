
from pathlib import Path
import cv2
import numpy as np
from measurement import measure_mask

# Part 1: Test known shapes
mask = np.zeros((224, 224), dtype=np.uint8)
mask[60:80, 50:90] = 255

r = measure_mask(mask)[0]
expected_length = (39 ** 2 + 19 ** 2) ** 0.5

checks = [
    ("area = 800 px", r["area_px"] == 800),
    (
        "bbox width/height = 40 x 20",
        (r["bbox_width_px"], r["bbox_height_px"]) == (40, 20)
    ),
    (
        "area percent = 1.59",
        abs(r["area_percent"] - 800 / 224 / 224 * 100) < 0.01
    ),
    (
        f"length = {expected_length:.2f}",
        abs(r["length_px"] - expected_length) < 0.01
    ),
    (
        "no mm values without a scale",
        "area_mm2" not in r
    ),
]

two = np.zeros((224, 224), dtype=np.uint8)
two[10:20, 10:20] = 255
two[100:120, 100:120] = 255

found = measure_mask(two)
checks.append((
    "two regions, largest first (400, then 100)",
    [x["area_px"] for x in found] == [400, 100]
))

scaled = measure_mask(mask, mm_per_pixel=0.1)[0]
checks.append((
    "with 0.1 mm/pixel: area = 8.0 mm2",
    abs(scaled["area_mm2"] - 8.0) < 1e-6
))

checks.append((
    "empty mask gives no regions",
    measure_mask(np.zeros((224, 224), np.uint8)) == []
))

print("Self-check:")
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")

# Part 2: Measure validation annotations
print("\nYour validation boxes (pixels only, no physical scale known):")
print(
    f"{'image':10s}{'regions':>8s}{'total px':>10s}"
    f"{'% image':>9s}{'largest bbox':>15s}{'length px':>11s}"
)

for mf in sorted(Path("data/annotations_val").glob("*_mask.png")):
    image_mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)

    if image_mask is None:
        raise ValueError(f"Could not read annotation: {mf}")

    regions = measure_mask(image_mask)

    total = sum(x["area_px"] for x in regions)
    big = regions[0] if regions else None

    if big is None:
        raise ValueError(f"Annotation has no white regions: {mf}")

    print(
        f"{mf.stem.removesuffix('_mask'):10s}"
        f"{len(regions):8d}{total:10d}"
        f"{100 * total / image_mask.size:8.1f}%"
        f"{str(big['bbox_width_px']) + ' x ' + str(big['bbox_height_px']):>15s}"
        f"{big['length_px']:11.1f}"
    )
