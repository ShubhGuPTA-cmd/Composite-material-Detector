
import csv

from evaluation import confusion_counts, precision_recall_f1
from preprocessing import load_gray
from detection import pipeline_score


CONFIGS = {
    "A baseline (gauss+CLAHE)": dict(),
    "B no CLAHE": dict(use_clahe=False),
    "C median blur": dict(blur="median"),
    "D bilateral blur": dict(blur="bilateral"),
    "E no blur": dict(blur="none"),
    "F bigger opening (5)": dict(open_size=5),
    "G min_area 100": dict(min_area=100),
}


def load_split(name):
    with open("data/splits.csv", newline="", encoding="utf-8") as fh:
        rows = [
            row for row in csv.DictReader(fh)
            if row["split"] == name
        ]

    images = [load_gray(row["path"]) for row in rows]
    labels = [
        0 if row["label"] == "negative" else 1
        for row in rows
    ]

    return images, labels


def metrics(scores, y_true, threshold):
    y_pred = [1 if score > threshold else 0 for score in scores]
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return precision_recall_f1(tp, fp, fn)


def best_threshold(scores, y_true):
    best_f1, best_t = -1.0, -1.0

    for threshold in [-1.0] + sorted(set(scores)):
        f1 = metrics(scores, y_true, threshold)[2]

        if f1 > best_f1:
            best_f1 = f1
            best_t = threshold

    return best_f1, best_t


# Load only train and validation data. Do not load test data.
train_imgs, y_train = load_split("train")
val_imgs, y_val = load_split("val")

print(f"train images: {len(train_imgs)}  val images: {len(val_imgs)}")

train_ref = metrics([1] * len(y_train), y_train, -1)[2]
val_ref = metrics([1] * len(y_val), y_val, -1)[2]

print(
    f"'flag everything' reference F1: "
    f"train={train_ref:.3f}  val={val_ref:.3f}\n"
)

header = (
    f"{'variant':28s}{'dark':6s}{'cutoff':>9s}"
    f"{'trainF1':>9s}{'valP':>7s}{'valR':>7s}{'valF1':>7s}"
)

print(header)
print("-" * len(header))

for name, cfg in CONFIGS.items():
    for dark in [True, False]:
        train_scores = [
            pipeline_score(image, dark_defects=dark, **cfg)
            for image in train_imgs
        ]

        val_scores = [
            pipeline_score(image, dark_defects=dark, **cfg)
            for image in val_imgs
        ]

        train_f1, cutoff = best_threshold(train_scores, y_train)
        precision, recall, val_f1 = metrics(
            val_scores, y_val, cutoff
        )

        print(
            f"{name:28s}{str(dark):6s}{cutoff:9.0f}"
            f"{train_f1:9.3f}{precision:7.3f}"
            f"{recall:7.3f}{val_f1:7.3f}"
        )
