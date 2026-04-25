"""
src/wtp_model.py
================
Willingness-to-Pay (WTP) model.

Since we don't have survey WTP labels in the UCI dataset, we engineer
a proxy WTP target from transaction behaviour:

    wtp_proxy = avg_order_value × log1p(frequency) × recency_weight

This captures the idea that:
  - Customers who spend more per order are willing to pay more
  - Customers who order more frequently have demonstrated higher engagement
  - Customers who bought recently are more relevant than lapsed ones

The GBR model then predicts this continuous WTP score from behavioural
and GMM soft-probability features.

SHAP explainability tells us which features drive WTP — this is the
part you explain in an interview: "The top driver was avg_order_value
(SHAP=0.38), followed by frequency and GMM segment probability."

Usage:
    from src.wtp_model import build_wtp_target, train_wtp_model, predict_wtp
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error


# ─────────────────────────────────────────────────────────────
# WTP proxy target
# ─────────────────────────────────────────────────────────────

def build_wtp_target(df: pd.DataFrame) -> pd.Series:
    """
    Engineer a WTP proxy from behavioural signals.

    Formula:
        recency_weight = 1 / (1 + recency_days/30)
        wtp = avg_order_value × log1p(frequency) × recency_weight

    This is a monotone function of all three RFM components:
      - Higher AOV → higher WTP
      - More orders → higher WTP (log-scaled to dampen outliers)
      - More recent → higher WTP (exponential decay)

    Returns:
        pd.Series "wtp_proxy", clipped to remove extreme outliers.
    """
    recency_weight = 1 / (1 + df["recency_days"] / 30)
    wtp = df["avg_order_value"] * np.log1p(df["frequency"]) * recency_weight

    # Clip at 99th percentile to remove extreme outliers
    cap = wtp.quantile(0.99)
    wtp = wtp.clip(upper=cap)

    print(f"  WTP proxy — min={wtp.min():.2f}  "
          f"median={wtp.median():.2f}  "
          f"max={wtp.max():.2f}")
    return wtp.rename("wtp_proxy")


# ─────────────────────────────────────────────────────────────
# Feature set
# ─────────────────────────────────────────────────────────────

def get_wtp_features(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature columns available in df for WTP modelling.
    Automatically includes GMM probability columns if present.
    """
    base = [
        "recency_days",
        "frequency",
        "monetary",
        "avg_order_value",
        "avg_items_per_order",
        "unique_products",
        "revenue_per_day",
        "active_days",
    ]
    gmm_cols = [c for c in df.columns if c.startswith("gmm_p")]
    return base + gmm_cols


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train_wtp_model(
    df: pd.DataFrame,
    target_col: str = "wtp_proxy",
    test_size: float = 0.20,
    random_state: int = 42,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.8,
) -> tuple[GradientBoostingRegressor, list[str], dict]:
    """
    Train a Gradient Boosting Regressor to predict WTP.

    Args:
        df            : Customer DataFrame with wtp_proxy column.
        target_col    : Name of the target column.
        test_size     : Held-out fraction.
        random_state  : Reproducibility seed.
        n_estimators  : Boosting rounds.
        learning_rate : Step shrinkage.
        max_depth     : Max tree depth.
        subsample     : Fraction of samples used per tree.

    Returns:
        (fitted_model, feature_list, metrics_dict)
    """
    feature_cols = get_wtp_features(df)
    # Drop any features missing from this df
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-4,
    )
    model.fit(X_train, y_train)

    y_pred    = model.predict(X_test)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    metrics = {
        "r2_test":    round(r2_score(y_test, y_pred), 4),
        "mae_test":   round(mean_absolute_error(y_test, y_pred), 4),
        "cv_r2_mean": round(cv_scores.mean(), 4),
        "cv_r2_std":  round(cv_scores.std(), 4),
        "n_estimators_used": model.n_estimators_,
    }

    print(f"  WTP Model — R²={metrics['r2_test']:.3f}  "
          f"MAE={metrics['mae_test']:.3f}  "
          f"CV-R²={metrics['cv_r2_mean']:.3f}±{metrics['cv_r2_std']:.3f}  "
          f"trees={metrics['n_estimators_used']}")

    return model, feature_cols, metrics


def predict_wtp(
    model: GradientBoostingRegressor,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """Predict WTP scores for all customers. Output is non-negative."""
    X    = df[feature_cols].fillna(0)
    pred = pd.Series(model.predict(X).clip(0), index=df.index, name="wtp_predicted")
    return pred.round(4)


# ─────────────────────────────────────────────────────────────
# SHAP explainability
# ─────────────────────────────────────────────────────────────

def compute_shap(
    model: GradientBoostingRegressor,
    df: pd.DataFrame,
    feature_cols: list[str],
    sample_n: int = 2000,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values for a random sample of customers.

    Args:
        model        : Fitted GBR.
        df           : Customer DataFrame.
        feature_cols : Feature column names.
        sample_n     : Sample size (full dataset is slow for SHAP).

    Returns:
        (shap_values array, X_sample DataFrame)
    """
    X_sample  = df[feature_cols].fillna(0).sample(
        min(sample_n, len(df)), random_state=random_state
    )
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_sample)
    return shap_vals, X_sample


def shap_importance(shap_vals: np.ndarray, X_sample: pd.DataFrame) -> pd.DataFrame:
    """
    Rank features by mean |SHAP value|.

    Returns DataFrame with columns: feature, mean_abs_shap.
    Sorted descending — most important feature first.
    """
    importance = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=X_sample.columns,
        name="mean_abs_shap",
    ).sort_values(ascending=False)

    return importance.reset_index().rename(columns={"index": "feature"})


def print_shap_bar(shap_vals: np.ndarray, X_sample: pd.DataFrame) -> None:
    """Print a text bar chart of SHAP feature importance to stdout."""
    imp = shap_importance(shap_vals, X_sample)
    max_val = imp["mean_abs_shap"].max()
    print("\n  SHAP Feature Importance:")
    for _, row in imp.iterrows():
        bar = "█" * int(row["mean_abs_shap"] / max_val * 30)
        print(f"    {row['feature']:<30} {row['mean_abs_shap']:.4f}  {bar}")
