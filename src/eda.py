"""
eda.py  -  Task 2: Exploratory data analysis and plots.

Produces three plots from the RAW (unstandardised) data:
  2.1  Ia, Ib, Ic vs row number (first 2000 rows), fault regions shaded.
  2.2  Scatter of Ia vs Ib, coloured by fault.
  2.3  Histogram of Ia for fault and no-fault rows.
"""

import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prepare import prepare_all


def plot_currents_vs_row(df, save_path="results/fig1_currents_vs_row.png"):
    """2.1 - Ia, Ib, Ic against row number for the first 2000 rows."""
    subset = df.head(2000)
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(subset.index, subset["Ia"], label="Ia", linewidth=0.7)
    ax.plot(subset.index, subset["Ib"], label="Ib", linewidth=0.7)
    ax.plot(subset.index, subset["Ic"], label="Ic", linewidth=0.7)

    # Shade fault regions
    fault = subset["fault"].values
    in_fault = False
    start = 0
    for i in range(len(fault)):
        if fault[i] == 1 and not in_fault:
            start = i
            in_fault = True
        elif fault[i] == 0 and in_fault:
            ax.axvspan(start, i, alpha=0.15, color="red")
            in_fault = False
    if in_fault:
        ax.axvspan(start, len(fault) - 1, alpha=0.15, color="red")

    ax.set_xlabel("Row number")
    ax.set_ylabel("Current (A)")
    ax.set_title("Line currents - first 2000 rows (red shading = fault)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")

    # Explanation
    print("""
  Observation: During fault intervals (shaded red) the currents jump to
  much larger absolute values. The three phase currents are sinusoidal in
  normal operation but become large and irregular during a fault, which is
  consistent with a short-circuit condition that drastically increases
  current flow.
""")


def plot_scatter_ia_ib(df, save_path="results/fig2_scatter_ia_ib.png"):
    """2.2 - Scatter plot of Ia vs Ib, coloured by fault."""
    fig, ax = plt.subplots(figsize=(6, 6))

    no_fault = df[df["fault"] == 0]
    fault    = df[df["fault"] == 1]

    ax.scatter(no_fault["Ia"], no_fault["Ib"], s=3, alpha=0.3,
               label="No fault", color="steelblue")
    ax.scatter(fault["Ia"], fault["Ib"], s=3, alpha=0.3,
               label="Fault", color="crimson")

    ax.set_xlabel("Ia (A)")
    ax.set_ylabel("Ib (A)")
    ax.set_title("Ia vs Ib coloured by fault")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")

    print("""
  Observation: The fault points (red) form a ring or annulus around the
  no-fault points (blue). The two classes overlap along any straight line
  through the origin, which means a linear classifier will not cleanly
  separate them. This is because faults increase the *magnitude* of
  current, but the *sign* (positive or negative) depends on the phase of
  the AC cycle.
""")


def plot_hist_ia(df, save_path="results/fig3_hist_ia.png"):
    """2.3 - Histogram of Ia for fault and no-fault rows."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(df.loc[df["fault"] == 0, "Ia"], bins=80, alpha=0.6,
            label="No fault", color="steelblue", density=True)
    ax.hist(df.loc[df["fault"] == 1, "Ia"], bins=80, alpha=0.6,
            label="Fault", color="crimson", density=True)

    ax.set_xlabel("Ia (A)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Ia by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")

    print("""
  Observation: The no-fault distribution is concentrated in a narrow band
  near zero, while the fault distribution has a much wider spread
  (heavier tails). However, the fault distribution is centred near zero
  too, confirming that a single threshold on Ia cannot separate the
  classes - you need to look at the magnitude (squared) of the currents.
""")


def answer_linear_separability():
    """One-paragraph answer about linear separability."""
    print("""
  Could a single straight line in the (Ia, Ib) plane separate the two
  classes?

  No. The scatter plot shows that fault points surround the no-fault
  points in a ring-like pattern. A fault simply increases the magnitude
  of the currents (|Ia|, |Ib| grow), but the sign depends on the AC
  phase at the instant of measurement. Therefore fault points appear at
  both large positive and large negative values, encircling the
  no-fault cluster near the origin. No single straight line can carve
  out this annular region - a non-linear decision boundary (or features
  that encode magnitude, such as I^2) is needed.
""")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    data = prepare_all()
    df = data["df_clean"]

    plot_currents_vs_row(df)
    plot_scatter_ia_ib(df)
    plot_hist_ia(df)
    answer_linear_separability()
