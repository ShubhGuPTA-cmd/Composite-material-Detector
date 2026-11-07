
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import CLASS_TO_INDEX, CompositeDataset
from model import build_model, predict_probs
from evaluation import auc


# Different lighting changes to test
PERTURBATIONS = {
    "original": lambda a: a,
    "darker (x0.5)": lambda a: a * 0.5,
    "brighter (x1.6)": lambda a: a * 1.6,
    "low contrast (x0.5)": lambda a: (
        (a - a.mean()) * 0.5 + a.mean()
    ),
    "gamma dark (1.8)": lambda a: a ** 1.8,
    "gamma bright (0.55)": lambda a: a ** 0.55,
    "shade left-right": lambda a: (
        a * np.linspace(0.4, 1.0, a.shape[1])[None, :, None]
    ),
}


# Saved models and whether neutral preprocessing is used
MODELS = {
    "NORMAL": ("models/resnet18_best.pt", False),
    "NEUTRAL": ("models/resnet18_neutral.pt", True),
}

# Select the split from the command-line argument.
# Defaults to validation if no argument is provided.
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"


class PerturbedDataset(CompositeDataset):
    """Dataset images with a lighting change applied."""

    def __init__(self, split, neutral, perturb):
        super().__init__(
            split,
            train=False,
            neutral=neutral,
        )
        self.perturb = perturb

    def __getitem__(self, index):
        row = self.rows[index]

        with Image.open(row["path"]) as im:
            image = im.convert("RGB")

        arr = np.asarray(image).astype(np.float32) / 255.0

        # Apply lighting change and keep pixels in the 0-1 range
        arr = np.clip(
            self.perturb(arr),
            0.0,
            1.0,
        )

        changed = Image.fromarray(
            (arr * 255).astype(np.uint8)
        )

        # Apply the usual model preprocessing
        return (
            self.transform(changed),
            CLASS_TO_INDEX[row["label"]],
        )


def evaluate(model, neutral, perturb):
    """Evaluate one model under one lighting condition."""

    dataset = PerturbedDataset(
        SPLIT,
        neutral,
        perturb,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    probs, labels = predict_probs(model, loader)

    probs = probs.numpy()
    labels = labels.numpy()

    # Overall 3-class accuracy
    acc = float(
        (probs.argmax(axis=1) == labels).mean()
    )

    # Task 1: all defects versus good
    s1 = 1 - probs[:, 0]

    auc1 = auc(
        s1[labels != 0].tolist(),
        s1[labels == 0].tolist(),
    )

    # Task 2: non-recoverable versus good
    keep = (labels == 0) | (labels == 2)

    s2 = probs[keep][:, 2]
    l2 = labels[keep]

    auc2 = auc(
        s2[l2 == 2].tolist(),
        s2[l2 == 0].tolist(),
    )

    return acc, auc1, auc2


# Load both saved models
loaded = {}

for name, (path, neutral) in MODELS.items():
    print(f"Loading {name} model from {path}...")

    model = build_model(pretrained=False)

    model.load_state_dict(
        torch.load(path, map_location="cpu")
    )

    model.eval()

    loaded[name] = (model, neutral)


# Print the selected split and results table
print()
print(f"Split: {SPLIT}")
print()

print(
    f"{'lighting change':22s}| "
    f"{'NORMAL acc':>10s} {'AUC1':>6s} {'AUC2':>6s} | "
    f"{'NEUTRAL acc':>11s} {'AUC1':>6s} {'AUC2':>6s}"
)

print("-" * 80)


for pname, perturb in PERTURBATIONS.items():
    row = []

    for name in ["NORMAL", "NEUTRAL"]:
        model, neutral = loaded[name]

        result = evaluate(
            model,
            neutral,
            perturb,
        )

        row.append(result)

    (a1, x1, y1), (a2, x2, y2) = row

    print(
        f"{pname:22s}| "
        f"{a1:10.3f}{x1:6.3f} {y1:6.3f} | "
        f"{a2:11.3f}{x2:6.3f} {y2:6.3f}"
    )


print()
print(
    "AUC1 = all defects vs good, "
    "AUC2 = non-recoverable vs good "
    f"({SPLIT} set)"
)
