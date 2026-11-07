
from collections import Counter

import torch

from dataset import CLASS_NAMES, make_loader
from model import build_model, predict_probs
from severity import LEVELS, assess


# ==================================================
# PART 1: SELF-CHECKS FOR SEVERITY RULES
# ==================================================

checks = []

checks.append((
    "clearly good -> None",
    assess([0.90, 0.05, 0.05])["severity"] == "None"
))

checks.append((
    "recoverable -> Low",
    assess([0.10, 0.80, 0.10])["severity"] == "Low"
))

checks.append((
    "unsure -> Medium",
    assess([0.10, 0.60, 0.30])["severity"] == "Medium"
))

checks.append((
    "non-recoverable -> High",
    assess([0.10, 0.20, 0.70])["severity"] == "High"
))

checks.append((
    "low confidence -> review flag",
    assess([0.40, 0.30, 0.30])["needs_review"]
))

checks.append((
    "size rule is off by default",
    assess(
        [0.10, 0.80, 0.10],
        region_area_percent=30
    )["severity"] == "Low"
))

checks.append((
    "size rule raises Low to Medium when switched on",
    assess(
        [0.10, 0.80, 0.10],
        region_area_percent=12.0,
        rules={"use_size_rule": True}
    )["severity"] == "Medium"
))

try:
    assess([0.5, 0.5, 0.5])
    checks.append(("bad probabilities are rejected", False))
except ValueError:
    checks.append(("bad probabilities are rejected", True))


print("Self-check:")

for name, ok in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name}")


# Stop if any self-check fails.
if not all(ok for _, ok in checks):
    raise RuntimeError(
        "A severity self-check failed. Fix it before evaluating the model."
    )


# ==================================================
# PART 2: LOAD THE NEUTRAL MODEL
# ==================================================

print("\nLoading neutral model...")

model = build_model(pretrained=False)

state_dict = torch.load(
    "models/resnet18_neutral.pt",
    map_location="cpu"
)

model.load_state_dict(state_dict)


# ==================================================
# PART 3: PREDICT ON VALIDATION IMAGES
# ==================================================

print("Evaluating validation images...")

model.eval()

probs, labels = predict_probs(
    model,
    make_loader("val", neutral=True)
)

probs = probs.numpy()
labels = labels.numpy()


# ==================================================
# PART 4: ASSIGN SEVERITY AND COUNT RESULTS
# ==================================================

counts = Counter()
review = Counter()
class_totals = Counter()

for p, y in zip(probs, labels):
    class_name = CLASS_NAMES[int(y)]

    result = assess([float(v) for v in p])

    severity = result["severity"]

    counts[(class_name, severity)] += 1
    class_totals[class_name] += 1

    if result["needs_review"]:
        review[class_name] += 1


# ==================================================
# PART 5: PRINT A CLEAR, ALIGNED TABLE
# ==================================================

print("\nSeverity on validation images")
print("(default rules, nothing tuned)")
print()

print(
    f"{'TRUE CLASS':28s}"
    f"{'NONE':>8s}"
    f"{'LOW':>8s}"
    f"{'MEDIUM':>8s}"
    f"{'HIGH':>8s}"
    f"{'REVIEW':>9s}"
    f"{'TOTAL':>8s}"
)

print("-" * 77)

for name in CLASS_NAMES:
    row = [
        counts[(name, level)]
        for level in LEVELS
    ]

    total = sum(row)

    print(
        f"{name:28s}"
        f"{row[0]:8d}"
        f"{row[1]:8d}"
        f"{row[2]:8d}"
        f"{row[3]:8d}"
        f"{review[name]:9d}"
        f"{total:8d}"
    )

print("-" * 77)

print(f"Total validation images: {len(labels)}")


# ==================================================
# PART 6: VERIFY ROW TOTALS
# ==================================================

print("\nRow-total verification:")

for name in CLASS_NAMES:
    total = class_totals[name]
    expected = 75

    if total == expected:
        print(f"  PASS  {name}: {total} images")
    else:
        print(
            f"  INFO  {name}: {total} images "
            f"(expected 75 for a balanced 225-image split)"
        )

print("\nStage 26 evaluation finished.")
print("Remember: severity is a dataset-label proxy,")
print("not a certified physical damage or safety assessment.")
