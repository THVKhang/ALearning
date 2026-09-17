"""Metrics for the model comparison: accuracy, macro-F1, parameter count."""

import numpy as np
from sklearn.metrics import f1_score


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def per_class_f1(y_true, y_pred, num_classes):
    scores = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
    return [float(score) for score in scores]


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
