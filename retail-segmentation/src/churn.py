"""
src/churn.py
============
Churn risk scoring — entirely derived from transaction behaviour,
no external churn labels needed.

The model weights three signals:
  Recency   (50%) — customers who haven't bought recently are at risk
  Frequency (30%) — low-order customers have less habit to break
  Monetary  (20%) — low spenders have less financial commitment

All three are ranked as percentiles so the score is robust to outliers
(a £50,000 whale doesn't skew the monetary signal).

Output: a score in [0, 1] — higher = more likely to churn.

Usage:
    from src.churn import compute_churn_risk, churn_summary
    customers["churn_risk"] = compute_churn_risk(customers)
"""

from __future__ import annotations
import pandas as pd
import numpy as np


# Default weights — must sum to 1.0
DEFAULT_WEIGHTS = {
    "recency_days": 0.50,
    "frequency":    0.30,
    "monetary":     0.20,
}


def compute_churn_risk(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Compute a churn risk score in [0.01, 0.99] for every customer.

    Args:
        df      : Customer feature DataFrame. Must have recency_days,
                  frequency, monetary columns.
        weights : Optional override of DEFAULT_WEIGHTS.

    Returns:
        pd.Series of churn risk scores, aligned with df.index.
    """
    w = weights or DEFAULT_WEIGHTS

    # Higher recency_days = hasn't bought recently = higher churn risk (no inversion)
    recency_risk   = df["recency_days"].rank(pct=True)

    # Lower frequency = fewer orders = higher churn risk (invert)
    frequency_risk = 1 - df["frequency"].rank(pct=True)

    # Lower monetary = less spend = higher churn risk (invert)
    monetary_risk  = 1 - df["monetary"].rank(pct=True)

    score = (
        recency_risk   * w["recency_days"] +
        frequency_risk * w["frequency"]    +
        monetary_risk  * w["monetary"]
    )
    return score.clip(0.01, 0.99).round(4)


def flag_high_risk(
    df: pd.DataFrame,
    risk_col: str = "churn_risk",
    threshold: float = 0.65,
) -> pd.DataFrame:
    """
    Return customers above the churn risk threshold, sorted by risk desc.

    Args:
        df        : Customer DataFrame with churn_risk column.
        risk_col  : Column name.
        threshold : Score above which a customer is "high risk".

    Returns:
        Filtered, sorted DataFrame.
    """
    high_risk = df[df[risk_col] >= threshold].sort_values(risk_col, ascending=False)
    print(f"  High-risk customers (≥{threshold}): {len(high_risk):,} "
          f"({len(high_risk)/len(df):.1%} of base)")
    return high_risk.reset_index(drop=True)


def churn_summary(
    df: pd.DataFrame,
    risk_col: str = "churn_risk",
    segment_col: str = "segment_name",
) -> pd.DataFrame:
    """
    Summarise churn risk by segment.

    Returns DataFrame with segment_name, n, avg_risk, pct_high_risk.
    """
    return (
        df.groupby(segment_col)
        .apply(lambda g: pd.Series({
            "n":               len(g),
            "avg_risk":        g[risk_col].mean().round(3),
            "pct_high_risk":   (g[risk_col] >= 0.65).mean().round(3),
            "avg_monetary":    g["monetary"].mean().round(2),
            "avg_recency":     g["recency_days"].mean().round(1),
        }))
        .reset_index()
        .sort_values("avg_risk", ascending=False)
    )
