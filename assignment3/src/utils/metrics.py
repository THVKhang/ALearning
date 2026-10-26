import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

CLASS_NAMES = [
    "Indoor-Level (00-10°)",
    "Indoor-Mid (20-30°)",
    "Indoor-High (40-50°)",
    "Outdoor-Level (00-10°)",
    "Outdoor-Mid (20-30°)",
    "Outdoor-High (40-50°)",
]

SHORT_CLASS_NAMES = [
    "Ind-Lvl",
    "Ind-Mid",
    "Ind-High",
    "Out-Lvl",
    "Out-Mid",
    "Out-High",
]


def compute_classification_metrics(y_true, y_pred):
    """Compute overall accuracy, macro-F1, per-class F1, domain accuracy, pitch accuracy, and confusion matrix."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_cls = f1_score(y_true, y_pred, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    # Sub-task 1: Environment Domain (Indoor {0,1,2} vs Outdoor {3,4,5})
    dom_true = (y_true >= 3).astype(np.int64)
    dom_pred = (y_pred >= 3).astype(np.int64)
    dom_acc = float((dom_true == dom_pred).mean())

    # Sub-task 2: Camera Elevation Regime (Level {0,3}, Mid {1,4}, High {2,5})
    elev_true = y_true % 3
    elev_pred = y_pred % 3
    elev_acc = float((elev_true == elev_pred).mean())

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "domain_accuracy": dom_acc,
        "elevation_accuracy": elev_acc,
        "per_class_f1": {CLASS_NAMES[i]: float(per_cls[i]) for i in range(len(CLASS_NAMES))},
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }
