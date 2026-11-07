import random
import numpy as np
def confusion_counts(y_true, y_pred):
    """y_true and y_pred are lists of 0 (good) or 1 (defect)."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    return tp, fp, fn, tn


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def auc(pos_scores, neg_scores):
    """Chance a random positive scores higher than a random negative.
    Ties count as half. 0.5 = no signal, 1.0 = perfect."""
    wins = 0.0

    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5

    return wins / (len(pos_scores) * len(neg_scores))


def balanced_accuracy(tp, fp, fn, tn):
    """Average of recall and specificity. 0.5 = no better than guessing."""
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (recall + specificity) / 2

def auc_interval(pos_scores, neg_scores, n_boot=200, seed=0):
    """Rough 95% range for AUC, by resampling images with replacement."""
    rng = random.Random(seed)
    values = []

    for _ in range(n_boot):
        p = [rng.choice(pos_scores) for _ in pos_scores]
        n = [rng.choice(neg_scores) for _ in neg_scores]
        values.append(auc(p, n))

    values.sort()

    return (
        values[int(0.025 * n_boot)],
        values[int(0.975 * n_boot) - 1],
    )


def iou_dice(pred_mask, true_mask):
    """Overlap between two masks (any value > 0 counts as marked)."""
    pred = pred_mask > 0
    true = true_mask > 0

    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    total = pred.sum() + true.sum()

    iou = inter / union if union > 0 else 1.0
    dice = 2 * inter / total if total > 0 else 1.0

    return float(iou), float(dice)

def best_cutoff_by_ba(scores, y_true):
    """Cutoff (flag if score > cutoff) with the best balanced accuracy on this data."""
    best_ba, best_cutoff = -1.0, 0.0
    for cutoff in sorted(set(scores)):
        y_pred = [1 if s > cutoff else 0 for s in scores]
        ba = balanced_accuracy(*confusion_counts(y_true, y_pred))
        if ba > best_ba:
            best_ba, best_cutoff = ba, cutoff
    return best_cutoff