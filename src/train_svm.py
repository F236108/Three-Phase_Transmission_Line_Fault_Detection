"""
train_svm.py  -  Task 4: Support Vector Machines.

Uses sklearn.svm.SVC on Set R (6 raw standardised features).
4.1  Linear SVM: C in {0.1, 1, 10, 100}
4.2  RBF SVM:    C in {0.1, 1, 10, 100} x gamma in {0.1, 1}
4.3  Results table
4.4  Save best linear and best RBF models with joblib
"""

import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prepare import prepare_all
from src.metrics import accuracy


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    data = prepare_all()

    X_train = data["X_train_R"]
    X_val   = data["X_val_R"]
    y_train = data["y_train"]
    y_val   = data["y_val"]

    Cs = [0.1, 1, 10, 100]
    gammas = [0.1, 1]

    results = []

    # -- 4.1 Linear SVM ---------------------------------------------------
    print("\n== Linear SVM ==")
    for C in Cs:
        model = SVC(kernel="linear", C=C)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy(y_val, y_pred)
        n_sv = int(model.n_support_.sum())
        results.append(("linear", C, "–", acc, n_sv, model))
        print(f"  C = {C:>5.1f}  ->  val acc = {acc:.4f},  support vectors = {n_sv}")

    # -- 4.2 RBF SVM ------------------------------------------------------
    print("\n== RBF SVM ==")
    for C in Cs:
        for gamma in gammas:
            model = SVC(kernel="rbf", C=C, gamma=gamma)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = accuracy(y_val, y_pred)
            n_sv = int(model.n_support_.sum())
            results.append(("rbf", C, gamma, acc, n_sv, model))
            print(f"  C = {C:>5.1f}, gamma = {gamma}  ->  val acc = {acc:.4f},  "
                  f"support vectors = {n_sv}")

    # -- 4.3 Table 3 (Section 4.3) ----------------------------------------
    with open("results/table3_svm.md", "w", encoding="utf-8") as f:
        f.write("# Table 3 – SVM Validation Results\n\n")
        f.write("| Kernel | C | γ | Validation accuracy | Support vectors |\n")
        f.write("|---|---|---|---|---|\n")
        for kernel, C, gamma, acc, n_sv, _ in results:
            f.write(f"| {kernel} | {C} | {gamma} | {acc:.4f} | {n_sv} |\n")
    print("\n  Saved results/table3_svm.md")

    # -- 4.4 Best models --------------------------------------------------
    linear_results = [r for r in results if r[0] == "linear"]
    rbf_results    = [r for r in results if r[0] == "rbf"]

    best_linear = max(linear_results, key=lambda r: r[3])
    best_rbf    = max(rbf_results,    key=lambda r: r[3])

    joblib.dump(best_linear[5], "results/best_linear_svm.joblib")
    joblib.dump(best_rbf[5],    "results/best_rbf_svm.joblib")

    print(f"\n  Best linear SVM: C = {best_linear[1]}, acc = {best_linear[3]:.4f}")
    print(f"  Best RBF SVM:    C = {best_rbf[1]}, gamma = {best_rbf[2]}, acc = {best_rbf[3]:.4f}")
    print("  Saved best_linear_svm.joblib, best_rbf_svm.joblib")

    # -- Figure: Validation accuracy vs C (log scale) ---------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    # Linear
    lin_Cs  = [r[1] for r in linear_results]
    lin_acc = [r[3] for r in linear_results]
    ax.plot(lin_Cs, lin_acc, "o-", label="Linear SVM")

    # RBF gamma = 0.1
    rbf_01  = [(r[1], r[3]) for r in rbf_results if r[2] == 0.1]
    ax.plot([c for c, a in rbf_01], [a for c, a in rbf_01], "s-",
            label="RBF SVM (γ=0.1)")

    # RBF gamma = 1
    rbf_1   = [(r[1], r[3]) for r in rbf_results if r[2] == 1]
    ax.plot([c for c, a in rbf_1], [a for c, a in rbf_1], "^-",
            label="RBF SVM (γ=1)")

    ax.set_xscale("log")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("SVM validation accuracy vs C")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/fig6_svm_accuracy_vs_C.png", dpi=150)
    plt.close(fig)
    print("  Saved results/fig6_svm_accuracy_vs_C.png")
