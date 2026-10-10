"""
train_logistic.py  -  Task 3: Train logistic regression models.

3.4  Choose alpha by running 2000 iterations for alpha in {0.01, 0.1, 1, 10}.
3.5  Train final models and report accuracy + theta.
"""

import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prepare import prepare_all
from src.logistic import sigmoid, hypothesis, cost, fit_logistic, predict
from src.metrics import accuracy


def add_bias(X):
    """Prepend a column of ones (bias term x0 = 1)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def choose_alpha(X_train, y_train, feature_set_name, alphas, n_iters=2000,
                 save_path=None):
    """
    Run gradient descent for each alpha, plot J vs iteration, and pick the
    largest alpha for which J decreases at every iteration.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    best_alpha = None

    for alpha in alphas:
        theta, history = fit_logistic(X_train, y_train, alpha, n_iters)
        ax.plot(range(1, n_iters + 1), history, label=f"α = {alpha}")

        # Check if J decreases at every step
        h = np.array(history)
        decreasing = np.all(np.diff(h) <= 0)
        if decreasing:
            best_alpha = alpha  # keep updating -> will end up with largest

        status_str = "monotone" if decreasing else "NOT monotone"
        print(f"  alpha = {alpha:5.2f}  ->  final J = {history[-1]:.6f}  [{status_str}]")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("J(θ)")
    ax.set_title(f"Cost vs iteration – {feature_set_name}")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved {save_path}")
    plt.close(fig)

    print(f"  -> Best alpha for {feature_set_name}: {best_alpha}")
    return best_alpha


def train_and_report(X_train, y_train, X_val, y_val, alpha, n_iters,
                     feature_set_name, feature_names):
    """Train the final model and report accuracy + theta."""
    theta, history = fit_logistic(X_train, y_train, alpha, n_iters)

    y_train_pred = predict(X_train, theta)
    y_val_pred   = predict(X_val, theta)
    train_acc = accuracy(y_train, y_train_pred)
    val_acc   = accuracy(y_val, y_val_pred)

    print(f"\n  {feature_set_name}:  alpha = {alpha},  final J = {history[-1]:.6f}")
    print(f"  Training accuracy:    {train_acc:.4f}")
    print(f"  Validation accuracy:  {val_acc:.4f}")
    print(f"  theta ({len(theta)} weights):")
    names = ["bias"] + list(feature_names)
    for name, val in zip(names, theta):
        print(f"    {name:>6s}: {val:+.6f}")

    return theta, history, train_acc, val_acc


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    data = prepare_all()

    # Add bias column
    X_train_R = add_bias(data["X_train_R"])
    X_val_R   = add_bias(data["X_val_R"])
    X_train_Q = add_bias(data["X_train_Q"])
    X_val_Q   = add_bias(data["X_val_Q"])

    y_train = data["y_train"]
    y_val   = data["y_val"]

    alphas = [0.01, 0.1, 1, 10]
    n_iters = 2000

    # -- 3.4 Choose alpha -------------------------------------------------
    print("\n== Choosing alpha for Set R ==")
    best_alpha_R = choose_alpha(X_train_R, y_train, "Set R", alphas, n_iters,
                                save_path="results/fig4_alpha_setR.png")

    print("\n== Choosing alpha for Set Q ==")
    best_alpha_Q = choose_alpha(X_train_Q, y_train, "Set Q", alphas, n_iters,
                                save_path="results/fig5_alpha_setQ.png")

    # -- 3.5 Train final models -------------------------------------------
    print("\n== Final logistic regression models ==")
    theta_R, hist_R, tr_acc_R, val_acc_R = train_and_report(
        X_train_R, y_train, X_val_R, y_val,
        best_alpha_R, n_iters, "Set R", data["feature_cols_R"]
    )
    theta_Q, hist_Q, tr_acc_Q, val_acc_Q = train_and_report(
        X_train_Q, y_train, X_val_Q, y_val,
        best_alpha_Q, n_iters, "Set Q", data["feature_cols_Q"]
    )

    # Save theta
    np.save("results/theta_R.npy", theta_R)
    np.save("results/theta_Q.npy", theta_Q)
    joblib.dump({"theta": theta_Q, "means": data["means_Q"], "stds": data["stds_Q"],
                 "feature_cols": data["feature_cols_Q"]},
                "results/logistic_Q_model.joblib")
    print("\n  Saved theta_R.npy, theta_Q.npy, logistic_Q_model.joblib")

    # -- Save Table 2 (Section 4.2) ---------------------------------------
    with open("results/table2_logistic.md", "w", encoding="utf-8") as f:
        f.write("# Table 2 – Logistic Regression\n\n")
        f.write("| Feature set | Chosen α | Final J(θ) | Validation accuracy |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Set R | {best_alpha_R} | {hist_R[-1]:.6f} | {val_acc_R:.4f} |\n")
        f.write(f"| Set Q | {best_alpha_Q} | {hist_Q[-1]:.6f} | {val_acc_Q:.4f} |\n")
    print("  Saved results/table2_logistic.md")

    # Verify: J starts at log(2) ~ 0.693
    J0_R = cost(X_train_R, y_train, np.zeros(X_train_R.shape[1]))
    J0_Q = cost(X_train_Q, y_train, np.zeros(X_train_Q.shape[1]))
    print(f"\n  Sanity check:  J(0) Set R = {J0_R:.6f},  J(0) Set Q = {J0_Q:.6f}  (should ≈ 0.693)")
