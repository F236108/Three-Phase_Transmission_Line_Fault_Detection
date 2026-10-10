"""
prepare.py  -  Task 1: Load, clean, split and standardise the data.

Steps
-----
1.1  Load raw CSV and print shape / columns / first rows.
1.2  Drop uninformative columns, rename the label to 'fault'.
1.3  Report fault / no-fault counts.
1.4  Split into train / val / test with the prescribed RNG seed.
1.5  Standardise using training-set statistics only.
"""

import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

# Allow imports from src/ when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.load_data import load_raw


# -- 1.1 Load raw data --------------------------------------------------------

def step_1_1(data_dir="data"):
    """Load the raw CSV and print diagnostics."""
    df = load_raw(data_dir)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head(5).to_string())
    return df


# -- 1.2 Clean columns --------------------------------------------------------

def step_1_2(df):
    """Drop uninformative columns and rename the label."""
    # Identify columns that carry no information (all NaN)
    cols_to_drop = [c for c in df.columns if df[c].isna().all()]
    print(f"\nColumns removed (all NaN): {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

    # Rename the label column
    df = df.rename(columns={"Output (S)": "fault"})

    # Confirm measurement columns are numeric
    measure_cols = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
    for c in measure_cols:
        assert pd.api.types.is_numeric_dtype(df[c]), f"{c} is not numeric!"

    # Missing values per column after cleaning
    print(f"\nMissing values per column after cleaning:")
    print(df.isnull().sum())

    return df, cols_to_drop


# -- 1.3 Class distribution ---------------------------------------------------

def step_1_3(df):
    """Print fault / no-fault counts."""
    counts = df["fault"].value_counts().sort_index()
    print(f"\nNo-fault (0): {counts[0]}")
    print(f"Fault   (1): {counts[1]}")
    return counts


# -- 1.4 Train / Val / Test split ---------------------------------------------

def split_data(df):
    """Split the DataFrame using the prescribed random generator."""
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_train, n_val = int(0.6 * len(df)), int(0.2 * len(df))
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        frac = split["fault"].mean()
        print(f"{name:5s}: {len(split):5d} rows,  fault fraction = {frac:.4f}")

    return df_train, df_val, df_test


# -- 1.5 Standardise ----------------------------------------------------------

def standardise(df_train, df_val, df_test, feature_cols):
    """
    Subtract training mean, divide by training std for each feature.
    Returns (X_train, X_val, X_test, means, stds) as NumPy arrays.
    """
    X_train_raw = df_train[feature_cols].values.astype(np.float64)
    X_val_raw   = df_val[feature_cols].values.astype(np.float64)
    X_test_raw  = df_test[feature_cols].values.astype(np.float64)

    means = X_train_raw.mean(axis=0)
    stds  = X_train_raw.std(axis=0)

    X_train = (X_train_raw - means) / stds
    X_val   = (X_val_raw   - means) / stds
    X_test  = (X_test_raw  - means) / stds

    return X_train, X_val, X_test, means, stds


def prepare_all(data_dir="data"):
    """
    Run the full pipeline and return everything downstream tasks need.

    Returns
    -------
    A dict with keys:
        df_raw, df_clean, df_train, df_val, df_test,
        cols_removed,
        feature_cols_R, feature_cols_Q,
        X_train_R, X_val_R, X_test_R,
        X_train_Q, X_val_Q, X_test_Q,
        y_train, y_val, y_test,
        means_R, stds_R, means_Q, stds_Q
    """
    df_raw = step_1_1(data_dir)
    df_clean, cols_removed = step_1_2(df_raw)
    counts = step_1_3(df_clean)

    print("\n-- Split --")
    df_train, df_val, df_test = split_data(df_clean)

    # Feature column names
    feature_cols_R = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

    # -- Set R (6 raw features, standardised) --
    X_train_R, X_val_R, X_test_R, means_R, stds_R = standardise(
        df_train, df_val, df_test, feature_cols_R
    )

    # -- Set Q (6 raw + 6 squared = 12 features) --
    # Compute squares from RAW values first, then standardise all 12 together
    sq_cols = ["Ia2", "Ib2", "Ic2", "Va2", "Vb2", "Vc2"]
    for split in (df_train, df_val, df_test):
        for raw, sq in zip(feature_cols_R, sq_cols):
            split[sq] = split[raw] ** 2

    feature_cols_Q = feature_cols_R + sq_cols
    X_train_Q, X_val_Q, X_test_Q, means_Q, stds_Q = standardise(
        df_train, df_val, df_test, feature_cols_Q
    )

    # Labels
    y_train = df_train["fault"].values
    y_val   = df_val["fault"].values
    y_test  = df_test["fault"].values

    print("\nPreparation complete.")
    print(f"Set R shape: train {X_train_R.shape}, val {X_val_R.shape}, test {X_test_R.shape}")
    print(f"Set Q shape: train {X_train_Q.shape}, val {X_val_Q.shape}, test {X_test_Q.shape}")

    # Generate and save Table 1 (Section 4.1)
    os.makedirs("results", exist_ok=True)
    table1_path = "results/table1_data_prep.md"
    missing_after = df_clean.isnull().sum().sum()
    with open(table1_path, "w", encoding="utf-8") as f:
        f.write("# Table 1 – Data Preparation\n\n")
        f.write("| Property | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| Raw rows / raw columns | {df_raw.shape[0]} rows / {df_raw.shape[1]} columns |\n")
        f.write(f"| Columns removed (names) | {', '.join(cols_removed)} |\n")
        f.write(f"| Missing values after cleaning | {missing_after} |\n")
        f.write(f"| Fault rows / no-fault rows | {counts.get(1, 0)} fault / {counts.get(0, 0)} no-fault |\n")
        f.write(f"| Train / validation / test rows | {len(df_train)} / {len(df_val)} / {len(df_test)} |\n")
        f.write(f"| Fault fraction in train / validation / test | {df_train['fault'].mean():.4f} / {df_val['fault'].mean():.4f} / {df_test['fault'].mean():.4f} |\n")
    print(f"Saved {table1_path}")

    return dict(
        df_raw=df_raw, df_clean=df_clean,
        cols_removed=cols_removed,
        df_train=df_train, df_val=df_val, df_test=df_test,
        feature_cols_R=feature_cols_R, feature_cols_Q=feature_cols_Q,
        X_train_R=X_train_R, X_val_R=X_val_R, X_test_R=X_test_R,
        X_train_Q=X_train_Q, X_val_Q=X_val_Q, X_test_Q=X_test_Q,
        y_train=y_train, y_val=y_val, y_test=y_test,
        means_R=means_R, stds_R=stds_R,
        means_Q=means_Q, stds_Q=stds_Q,
    )


if __name__ == "__main__":
    data = prepare_all()
