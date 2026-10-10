"""
metrics.py  -  Task 5 helpers: accuracy and confusion matrix (from scratch).

No sklearn.metrics is used.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np


def accuracy(y_true, y_pred):
    """
    Accuracy = (1/m) * sum( y_pred == y_true ).

    Parameters
    ----------
    y_true : array-like, shape (m,)
    y_pred : array-like, shape (m,)

    Returns
    -------
    float - fraction of correct predictions
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred):
    """
    2x2 confusion matrix for binary classification.

    Returns
    -------
    dict with keys: TN, FP (false alarm), FN (missed fault), TP
    Also returns a 2x2 NumPy array [[TN, FP], [FN, TP]].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))   # false alarm
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))   # missed fault
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))

    cm = np.array([[TN, FP], [FN, TP]])

    return {
        "TN": TN,
        "FP": FP,   # false alarm
        "FN": FN,   # missed fault
        "TP": TP,
        "matrix": cm,
    }


def print_confusion_matrix(cm_dict, model_name="Model"):
    """Pretty-print a confusion matrix dictionary."""
    print(f"\n  Confusion matrix for {model_name}:")
    print(f"                 Predicted 0   Predicted 1")
    print(f"    Actual 0     {cm_dict['TN']:>8d}      {cm_dict['FP']:>8d}   (false alarms: {cm_dict['FP']})")
    print(f"    Actual 1     {cm_dict['FN']:>8d}      {cm_dict['TP']:>8d}   (missed faults: {cm_dict['FN']})")
    total = cm_dict["TN"] + cm_dict["FP"] + cm_dict["FN"] + cm_dict["TP"]
    print(f"    Total: {total}")
