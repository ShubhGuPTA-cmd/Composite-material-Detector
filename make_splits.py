
import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/raw/carbon-multiclass")
OUT_FILE = Path("data/splits.csv")

CLASSES = [
    "negative",
    "recoverable_defects",
    "non_recoverable_defects",
]

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SEED = 42


def file_hash(path):
    """Return a fingerprint of the file contents."""
    return hashlib.md5(path.read_bytes()).hexdigest()


rng = random.Random(SEED)
rows = []
seen = {}

for name in CLASSES:
    class_dir = DATA_DIR / name
    files = sorted(
        f for f in class_dir.rglob("*")
        if f.is_file()
        and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not files:
        raise SystemExit(f"No images found in {class_dir}")

    groups = {}

    for f in files:
        h = file_hash(f)

        if h in seen and seen[h] != name:
            print(
                f"WARNING: {f.name} matches an image "
                f"in class {seen[h]}"
            )

        seen[h] = name
        groups.setdefault(h, []).append(f)

    print(
        f"{name}: {len(files)} files, "
        f"{len(files) - len(groups)} exact duplicates"
    )

    group_list = list(groups.values())
    rng.shuffle(group_list)

    total = len(files)
    count = 0

    for group in group_list:
        if count < TRAIN_FRAC * total:
            split = "train"
        elif count < (TRAIN_FRAC + VAL_FRAC) * total:
            split = "val"
        else:
            split = "test"

        for f in group:
            rows.append((f.as_posix(), name, split))

        count += len(group)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUT_FILE.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["path", "label", "split"])
    writer.writerows(rows)

print("\nImages per class and split:")
counts = Counter(
    (label, split) for _, label, split in rows
)

for name in CLASSES:
    print(
        f"{name:25s}",
        {
            split: counts[(name, split)]
            for split in ["train", "val", "test"]
        },
    )

print("\nSaved:", OUT_FILE)
