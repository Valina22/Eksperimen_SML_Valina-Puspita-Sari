"""
=============================================================
automate_Valina-Puspita-Sari.py
Automated Preprocessing Pipeline — Heart Failure Prediction
Nama Siswa : Valina Puspita Sari
Dataset    : Heart Failure Prediction (fedesoriano, Kaggle)
=============================================================
"""

import os
import sys
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure root logger with file + console handlers."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"preprocessing_{ts}.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(funcName)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("smsml_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logging()


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe.
    """
    logger.info(f"Loading dataset from: {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded — shape: {df.shape}")
        logger.info(f"Columns : {list(df.columns)}")
        logger.debug(f"First 3 rows:\n{df.head(3).to_string()}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as exc:
        logger.error(f"Unexpected error loading data: {exc}")
        raise


# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
def perform_eda(df: pd.DataFrame, output_dir: str = "eda_output") -> dict:
    """
    Perform full Exploratory Data Analysis and save plots.

    Parameters
    ----------
    df         : raw dataframe
    output_dir : directory to store EDA plots

    Returns
    -------
    dict with EDA summary statistics
    """
    logger.info("═" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("═" * 60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Basic info ---
    logger.info(f"Shape           : {df.shape}")
    logger.info(f"Total NaN       : {df.isnull().sum().sum()}")
    logger.info(f"Duplicate rows  : {df.duplicated().sum()}")
    logger.info(f"Dtypes:\n{df.dtypes.to_string()}")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "HeartDisease" in num_cols:
        num_cols.remove("HeartDisease")

    # --- Descriptive stats ---
    desc = df[num_cols].describe()
    logger.info(f"\nDescriptive Statistics:\n{desc.to_string()}")

    # === Plot 1: Distribution of numerical features ===
    n = len(num_cols)
    ncols_fig = 3
    nrows_fig = (n + ncols_fig - 1) // ncols_fig
    fig, axes = plt.subplots(nrows_fig, ncols_fig, figsize=(15, 4 * nrows_fig))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=30, color="#534AB7", edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Distribusi Fitur Numerik", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_num_distributions.png", dpi=120)
    plt.close()
    logger.info("Saved: 01_num_distributions.png")

    # === Plot 2: Categorical features ===
    if cat_cols:
        ncat = len(cat_cols)
        fig, axes = plt.subplots(1, ncat, figsize=(5 * ncat, 4))
        if ncat == 1:
            axes = [axes]
        for i, col in enumerate(cat_cols):
            vc = df[col].value_counts()
            axes[i].bar(vc.index, vc.values, color="#1D9E75", edgecolor="white")
            axes[i].set_title(col, fontsize=11)
            axes[i].set_xlabel("Category")
            axes[i].set_ylabel("Count")
            axes[i].tick_params(axis="x", rotation=30)
        plt.suptitle("Distribusi Fitur Kategorikal", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/02_cat_distributions.png", dpi=120)
        plt.close()
        logger.info("Saved: 02_cat_distributions.png")

    # === Plot 3: Target distribution ===
    fig, ax = plt.subplots(figsize=(5, 4))
    vc = df["HeartDisease"].value_counts()
    ax.bar(["No Disease (0)", "Heart Disease (1)"], vc.values,
           color=["#1D9E75", "#D85A30"], edgecolor="white")
    for rect, val in zip(ax.patches, vc.values):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 5,
                f"{val}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=10)
    ax.set_title("Distribusi Target (HeartDisease)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_target_distribution.png", dpi=120)
    plt.close()
    logger.info("Saved: 03_target_distribution.png")

    # === Plot 4: Correlation heatmap ===
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in cat_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    corr = df_encoded.corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                annot_kws={"size": 9})
    ax.set_title("Correlation Heatmap (all features)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_correlation_heatmap.png", dpi=120)
    plt.close()
    logger.info("Saved: 04_correlation_heatmap.png")

    # === Plot 5: Boxplots for outlier detection ===
    fig, axes = plt.subplots(nrows_fig, ncols_fig, figsize=(15, 4 * nrows_fig))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#CECBF6", color="#534AB7"),
                        medianprops=dict(color="#D85A30", linewidth=2))
        axes[i].set_title(col, fontsize=11)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Boxplot — Outlier Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_boxplots.png", dpi=120)
    plt.close()
    logger.info("Saved: 05_boxplots.png")

    # === Plot 6: Bivariate — numerical vs target ===
    fig, axes = plt.subplots(nrows_fig, ncols_fig, figsize=(15, 4 * nrows_fig))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        for tval, color in [(0, "#1D9E75"), (1, "#D85A30")]:
            axes[i].hist(df[df["HeartDisease"] == tval][col].dropna(),
                         bins=25, alpha=0.6, color=color,
                         label=f"{'No' if tval==0 else 'Yes'} HD", edgecolor="white")
        axes[i].set_title(col, fontsize=11)
        axes[i].legend(fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Distribusi Numerik per Kelas Target", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_bivariate_num_target.png", dpi=120)
    plt.close()
    logger.info("Saved: 06_bivariate_num_target.png")

    # === Summary ===
    summary = {
        "shape"       : df.shape,
        "missing"     : int(df.isnull().sum().sum()),
        "duplicates"  : int(df.duplicated().sum()),
        "num_cols"    : num_cols,
        "cat_cols"    : cat_cols,
        "target_dist" : df["HeartDisease"].value_counts().to_dict(),
    }
    logger.info(f"EDA Summary: {summary}")
    return summary


# ─────────────────────────────────────────────
# 3. MISSING VALUES
# ─────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - Numerical  → median imputation (robust to outliers)
    - Categorical → mode imputation
    """
    logger.info("Handling missing values …")
    df = df.copy()

    missing_before = df.isnull().sum().sum()
    logger.info(f"Missing values before: {missing_before}")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"  {col}: filled {n_missing} NaN with median={median_val:.3f}")

    for col in cat_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info(f"  {col}: filled {n_missing} NaN with mode='{mode_val}'")

    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values after : {missing_after}")
    return df


# ─────────────────────────────────────────────
# 4. DUPLICATE HANDLING
# ─────────────────────────────────────────────
def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows, keeping first occurrence."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df)
    logger.info(f"Duplicates removed: {n_before - n_after} rows "
                f"({n_before} → {n_after})")
    return df


# ─────────────────────────────────────────────
# 5. OUTLIER HANDLING
# ─────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame, method: str = "iqr",
                    factor: float = 1.5) -> pd.DataFrame:
    """
    Handle outliers in numerical columns.

    method : 'iqr' (capping) or 'zscore' (remove rows with |z|>3)
    factor : IQR multiplier (default 1.5)
    """
    logger.info(f"Handling outliers — method='{method}', factor={factor}")
    df = df.copy()

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "HeartDisease" in num_cols:
        num_cols.remove("HeartDisease")

    n_before = len(df)

    if method == "iqr":
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower=lower, upper=upper)
            logger.info(f"  {col}: {n_outliers} outliers capped "
                        f"[{lower:.2f}, {upper:.2f}]")

    elif method == "zscore":
        z_scores = np.abs(stats.zscore(df[num_cols]))
        mask = (z_scores < 3).all(axis=1)
        df = df[mask].reset_index(drop=True)
        logger.info(f"  Z-score: removed {n_before - len(df)} rows")

    else:
        logger.warning(f"Unknown method '{method}', skipping outlier handling")

    logger.info(f"Shape after outlier handling: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 6. ENCODING
# ─────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features:
    - Binary categories (2 unique values)  → Label Encoding
    - Multi-class categories               → One-Hot Encoding
    """
    logger.info("Encoding categorical features …")
    df = df.copy()

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()

    ohe_cols = []
    label_cols = []

    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique == 2:
            df[col] = le.fit_transform(df[col])
            label_cols.append(col)
            logger.info(f"  Label Encoded: {col} ({le.classes_})")
        else:
            ohe_cols.append(col)
            logger.info(f"  OneHot queued: {col} ({n_unique} unique)")

    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=int)
        logger.info(f"  OneHot applied to: {ohe_cols}")

    logger.info(f"Shape after encoding: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# 7. FEATURE SCALING
# ─────────────────────────────────────────────
def scale_features(df: pd.DataFrame,
                   scaler_type: str = "standard") -> tuple[pd.DataFrame, object]:
    """
    Scale numerical features.

    scaler_type : 'standard' (StandardScaler, default)
                  'minmax'   (MinMaxScaler)
                  'robust'   (RobustScaler)

    Returns (scaled_df, fitted_scaler)
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler
    logger.info(f"Scaling features — scaler='{scaler_type}'")

    df = df.copy()
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if "HeartDisease" in num_cols:
        num_cols.remove("HeartDisease")

    scaler_map = {
        "standard": StandardScaler(),
        "minmax"  : MinMaxScaler(),
        "robust"  : RobustScaler(),
    }
    scaler = scaler_map.get(scaler_type, StandardScaler())
    df[num_cols] = scaler.fit_transform(df[num_cols])
    logger.info(f"Scaled {len(num_cols)} numerical features: {num_cols}")
    return df, scaler


# ─────────────────────────────────────────────
# 8. FEATURE SELECTION
# ─────────────────────────────────────────────
def feature_selection(df: pd.DataFrame,
                       target_col: str = "HeartDisease",
                       threshold: float = 0.01,
                       output_dir: str = "eda_output") -> pd.DataFrame:
    """
    Select features using Mutual Information.
    Drops features with MI score below threshold.
    """
    logger.info(f"Feature selection — MI threshold={threshold}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    logger.info(f"Mutual Information scores:\n{mi_series.to_string()}")

    # Plot MI
    fig, ax = plt.subplots(figsize=(10, 5))
    mi_series.plot(kind="barh", ax=ax, color="#534AB7", edgecolor="white")
    ax.axvline(x=threshold, color="#D85A30", linestyle="--", label=f"Threshold={threshold}")
    ax.set_title("Mutual Information — Feature Importance", fontsize=12, fontweight="bold")
    ax.set_xlabel("MI Score")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_mutual_information.png", dpi=120)
    plt.close()
    logger.info("Saved: 07_mutual_information.png")

    # Keep features above threshold
    selected = mi_series[mi_series >= threshold].index.tolist()
    dropped  = mi_series[mi_series < threshold].index.tolist()

    logger.info(f"Selected ({len(selected)}): {selected}")
    logger.info(f"Dropped  ({len(dropped)}) : {dropped}")

    df_selected = df[selected + [target_col]]
    logger.info(f"Shape after feature selection: {df_selected.shape}")
    return df_selected


# ─────────────────────────────────────────────
# 9. DATA SPLITTING
# ─────────────────────────────────────────────
def split_data(df: pd.DataFrame,
               target_col: str = "HeartDisease",
               test_size: float = 0.2,
               val_size: float = 0.1,
               random_state: int = 42) -> tuple:
    """
    Stratified split: 70% train / 10% val / 20% test

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info(f"Splitting data — test={test_size}, val={val_size}, "
                f"random_state={random_state}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First: split train+val vs test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second: split train vs val from remaining
    adjusted_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=adjusted_val, stratify=y_tv, random_state=random_state
    )

    logger.info(f"  Train : {X_train.shape[0]} rows ({X_train.shape[0]/len(df)*100:.1f}%)")
    logger.info(f"  Val   : {X_val.shape[0]}   rows ({X_val.shape[0]/len(df)*100:.1f}%)")
    logger.info(f"  Test  : {X_test.shape[0]}  rows ({X_test.shape[0]/len(df)*100:.1f}%)")

    for split, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        dist = y_split.value_counts(normalize=True).round(3).to_dict()
        logger.info(f"  {split} target dist: {dist}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# 10. SAVE DATASET
# ─────────────────────────────────────────────
def save_dataset(X_train, X_val, X_test, y_train, y_val, y_test,
                 output_dir: str = "preprocessing/dataset_preprocessing") -> None:
    """
    Save preprocessed splits as CSV files:
        train.csv | val.csv | test.csv
    Each file contains features + target column appended.
    """
    logger.info(f"Saving preprocessed datasets to: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy(); train_df["HeartDisease"] = y_train.values
    val_df   = X_val.copy();   val_df["HeartDisease"]   = y_val.values
    test_df  = X_test.copy();  test_df["HeartDisease"]  = y_test.values

    train_path = f"{output_dir}/train.csv"
    val_path   = f"{output_dir}/val.csv"
    test_path  = f"{output_dir}/test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)

    logger.info(f"  Saved: {train_path} — {train_df.shape}")
    logger.info(f"  Saved: {val_path}   — {val_df.shape}")
    logger.info(f"  Saved: {test_path}  — {test_df.shape}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main(args=None):
    """Entry point — orchestrates the full preprocessing pipeline."""

    parser = argparse.ArgumentParser(
        description="SMSML Automated Preprocessing — Valina Puspita Sari",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str,
        default="dataset_raw/heart.csv",
        help="Path to raw CSV dataset",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="preprocessing/dataset_preprocessing",
        help="Directory to save preprocessed CSV splits",
    )
    parser.add_argument(
        "--eda_dir", type=str,
        default="preprocessing/eda_output",
        help="Directory to save EDA plots",
    )
    parser.add_argument(
        "--scaler", type=str,
        default="standard",
        choices=["standard", "minmax", "robust"],
        help="Feature scaling method",
    )
    parser.add_argument(
        "--outlier_method", type=str,
        default="iqr",
        choices=["iqr", "zscore"],
        help="Outlier handling method",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Fraction of data for test split",
    )
    parser.add_argument(
        "--val_size", type=float, default=0.1,
        help="Fraction of data for validation split",
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip_eda", action="store_true",
        help="Skip EDA step (faster for CI/CD)",
    )

    cfg = parser.parse_args(args)

    logger.info("=" * 60)
    logger.info("SMSML PREPROCESSING PIPELINE STARTED")
    logger.info(f"Config: {vars(cfg)}")
    logger.info("=" * 60)

    try:
        # Step 1: Load
        df = load_data(cfg.input)

        # Step 2: EDA
        if not cfg.skip_eda:
            eda_summary = perform_eda(df, output_dir=cfg.eda_dir)
        else:
            logger.info("EDA step skipped (--skip_eda flag)")

        # Step 3: Missing values
        df = handle_missing_values(df)

        # Step 4: Duplicates
        df = handle_duplicates(df)

        # Step 5: Outliers
        df = handle_outliers(df, method=cfg.outlier_method)

        # Step 6: Encoding
        df = encode_features(df)

        # Step 7: Feature selection
        df = feature_selection(df, output_dir=cfg.eda_dir)

        # Step 8: Scaling
        df, scaler = scale_features(df, scaler_type=cfg.scaler)

        # Step 9: Split
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.random_state,
        )

        # Step 10: Save
        save_dataset(X_train, X_val, X_test, y_train, y_val, y_test,
                     output_dir=cfg.output_dir)

        logger.info("=" * 60)
        logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY ✓")
        logger.info("=" * 60)

    except Exception as exc:
        logger.exception(f"Pipeline failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
