
from collections import Counter
import matplotlib.pyplot as plt
import torch

from dataset import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    CompositeDataset,
    make_loader,
)

# 1. Count images in the training and validation splits
for split in ["train", "val"]:
    ds = CompositeDataset(split)
    counts = Counter(r["label"] for r in ds.rows)

    print(
        split,
        len(ds),
        {name: counts[name] for name in CLASS_NAMES}
    )

# 2. Confirm that the test split is protected
try:
    make_loader("test")
except ValueError as err:
    print("Test-split guard works:", err)

# 3. Load one batch of 32 training images
loader = make_loader("train", batch_size=32)
images, labels = next(iter(loader))

print(
    "batch images:", tuple(images.shape),
    " labels:", tuple(labels.shape)
)

print(
    "pixel mean / std after normalizing:",
    round(images.mean().item(), 2),
    round(images.std().item(), 2)
)

# 4. Undo normalization to display natural image colours
mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))

for ax, img, lab in zip(
    axes.ravel(), images, labels
):
    shown = (
        (img * std + mean)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )

    ax.imshow(shown)
    ax.set_title(CLASS_NAMES[lab.item()], fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()
