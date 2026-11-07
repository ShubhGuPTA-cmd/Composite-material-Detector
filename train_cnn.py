
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from dataset import CLASS_NAMES, make_loader
from model import build_model, predict_probs
from evaluation import auc, auc_interval, balanced_accuracy

# Normal mode: python src\train_cnn.py
# Neutral mode: python src\train_cnn.py neutral
NEUTRAL = len(sys.argv) > 1 and sys.argv[1] == "neutral"

EPOCHS = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

torch.manual_seed(0)
Path("models").mkdir(exist_ok=True)

MODEL_FILE = Path(
    "models/resnet18_neutral.pt"
    if NEUTRAL
    else "models/resnet18_best.pt"
)

print(
    "Mode:",
    "NEUTRAL (grayscale, brightness and contrast removed)"
    if NEUTRAL else "normal",
)

train_loader = make_loader(
    "train", BATCH_SIZE, neutral=NEUTRAL
)
val_loader = make_loader(
    "val", BATCH_SIZE, neutral=NEUTRAL
)

model = build_model()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)


best_acc = -1.0

for epoch in range(1, EPOCHS + 1):
    start = time.perf_counter()
    model.train()

    total_loss, seen = 0.0, 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        seen += len(labels)

    probs, labels = predict_probs(model, val_loader)
    val_acc = (
        (probs.argmax(dim=1) == labels)
        .float()
        .mean()
        .item()
    )

    elapsed = time.perf_counter() - start

    print(
        f"epoch {epoch}/{EPOCHS}  "
        f"train loss {total_loss / seen:.3f}  "
        f"val accuracy {val_acc:.3f}  "
        f"({elapsed:.0f} s)"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_FILE)
        print("   saved as best so far")


# Load the best validation checkpoint
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
probs, labels = predict_probs(model, val_loader)

probs = probs.numpy()
labels = labels.numpy()
pred = probs.argmax(axis=1)

print("\nConfusion matrix on val (rows = true class, columns = predicted):")
print(f"{'':25s}", [n[:8] for n in CLASS_NAMES])

for i, name in enumerate(CLASS_NAMES):
    print(
        f"{name:25s}",
        [
            int(((labels == i) & (pred == j)).sum())
            for j in range(3)
        ],
    )


def report(title, scores, y_true, y_pred, reference):
    pos = scores[y_true == 1].tolist()
    neg = scores[y_true == 0].tolist()

    a = auc(pos, neg)
    lo, hi = auc_interval(pos, neg)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    ba = balanced_accuracy(tp, fp, fn, tn)

    print(f"\n{title}")
    print(
        f"  CNN:          AUC={a:.3f} "
        f"(95% range {lo:.2f}-{hi:.2f})  "
        f"balanced accuracy={ba:.3f}  "
        f"precision={precision:.3f}  recall={recall:.3f}"
    )
    print(f"  edge_density: {reference}")


# Task 1: all defects vs good
y1 = (labels != 0).astype(int)

report(
    f"Task 1: all defects vs good "
    f"(val: {int(y1.sum())} defect / "
    f"{int((1 - y1).sum())} good)",
    1 - probs[:, 0],
    y1,
    (pred != 0).astype(int),
    "AUC=0.719 (95% range 0.64-0.79)  "
    "balanced accuracy=0.680",
)


# Task 2: non-recoverable defects vs good
sel = (labels == 0) | (labels == 2)
y2 = (labels[sel] == 2).astype(int)

report(
    f"Task 2: non-recoverable vs good "
    f"(val: {int(y2.sum())} defect / "
    f"{int((1 - y2).sum())} good)",
    probs[sel][:, 2],
    y2,
    (pred[sel] == 2).astype(int),
    "AUC=0.811 (95% range 0.74-0.86)  "
    "balanced accuracy=0.707  "
    "precision=0.754  recall=0.613",
)
