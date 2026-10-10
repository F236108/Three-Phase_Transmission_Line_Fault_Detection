"""
test_models.py  -  Task 5: Evaluate all four final models on the TEST set (once).

Models:
  1. Logistic regression on Set R
  2. Logistic regression on Set Q
  3. Best linear SVM (on Set R)
  4. Best RBF SVM (on Set R)
"""

import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prepare import prepare_all
from src.logistic import predict as logistic_predict
from src.metrics import accuracy, confusion_matrix, print_confusion_matrix


def add_bias(X):
    """Prepend a column of ones (bias term x0 = 1)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    data = prepare_all()

    X_test_R = data["X_test_R"]
    X_test_Q = data["X_test_Q"]
    y_test   = data["y_test"]

    # Load saved models
    theta_R = np.load("results/theta_R.npy")
    theta_Q = np.load("results/theta_Q.npy")
    best_linear_svm = joblib.load("results/best_linear_svm.joblib")
    best_rbf_svm    = joblib.load("results/best_rbf_svm.joblib")

    # -- Predictions ------------------------------------------------------
    models = [
        ("Logistic regression, Set R", logistic_predict(add_bias(X_test_R), theta_R)),
        ("Logistic regression, Set Q", logistic_predict(add_bias(X_test_Q), theta_Q)),
        (f"Linear SVM (C={best_linear_svm.C})",   best_linear_svm.predict(X_test_R)),
        (f"RBF SVM (C={best_rbf_svm.C}, γ={best_rbf_svm.gamma})",
                                                    best_rbf_svm.predict(X_test_R)),
    ]

    # -- Report -----------------------------------------------------------
    print("\n== Test Results ==")
    table_rows = []

    for name, y_pred in models:
        acc = accuracy(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print_confusion_matrix(cm, name)
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Missed faults: {cm['FN']}")
        table_rows.append((name, acc, cm["TN"], cm["FP"], cm["FN"], cm["TP"]))

    # -- Table 4 (Section 4.4) --------------------------------------------
    with open("results/table4_test.md", "w", encoding="utf-8") as f:
        f.write("# Table 4 – Test Results\n\n")
        f.write("| Model | Accuracy | TN | False alarms | Missed faults | TP |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, acc, tn, fp, fn, tp in table_rows:
            f.write(f"| {name} | {acc:.4f} | {tn} | {fp} | {fn} | {tp} |\n")
    print("\n  Saved results/table4_test.md")

    # -- Table 5: Learned theta (Set Q) (Section 4.5) ---------------------
    feature_names_Q = ["bias", "Ia", "Ib", "Ic", "Va", "Vb", "Vc",
                       "Ia²", "Ib²", "Ic²", "Va²", "Vb²", "Vc²"]
    with open("results/table5_theta_Q.md", "w", encoding="utf-8") as f:
        f.write("# Table 5 – Learned θ (Set Q)\n\n")
        f.write("| Weight | Feature | Value |\n")
        f.write("|---|---|---|\n")
        for i, (name, val) in enumerate(zip(feature_names_Q, theta_Q)):
            weight_label = f"θ{i}"
            f.write(f"| {weight_label} | {name} | {val:+.6f} |\n")
    print("  Saved results/table5_theta_Q.md")
