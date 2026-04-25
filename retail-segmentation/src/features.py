"""
src/features.py
===============
Builds the customer-level feature table from the cleaned transaction log.

Every row in the output = one customer.
Features built:
  RFM      — recency_days, frequency, monetary
  Derived  — avg_order_value, avg_items_per_order, unique_products,
              unique_invoices, revenue_per_day, active_days,
              cancellation_rate (from raw), country, top_country
  Scores   — r_score, f_score, m_score (1-5 quintiles), rfm_total

Usage:
    from src.features import build_customer_features
    customers = build_customer_features(df_clean)
"""

import pandas as pd
import numpy as np
from datetime import timedelta


# ─────────────────────────────────────────────────────────────
# Core builder
# ─────────────────────────────────────────────────────────────

def build_customer_features(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Aggregate a cleaned transaction DataFrame into one row per customer.

    Args:
        df:            Cleaned DataFrame from data/clean_data.py.
                       Must have: customer_id, invoice_no, invoice_date,
                                  quantity, unit_price, revenue, country.
        snapshot_date: The "today" for recency calculation.
                       Defaults to max(invoice_date) + 1 day.

    Returns:
        customer_features DataFrame indexed by customer_id.
    """
    if snapshot_date is None:
        snapshot_date = df["invoice_date"].max() + timedelta(days=1)

    print(f"  Snapshot date  : {snapshot_date.date()}")
    print(f"  Transactions   : {len(df):,}")
    print(f"  Customers      : {df['customer_id'].nunique():,}")

    # ── RFM ────────────────────────────────────────────────
    rfm = (
        df.groupby("customer_id")
        .agg(
            last_purchase   = ("invoice_date",  "max"),
            frequency       = ("invoice_no",    "nunique"),   # unique orders
            monetary        = ("revenue",        "sum"),
            total_items     = ("quantity",       "sum"),
            unique_products = ("stock_code",     "nunique"),
            n_countries     = ("country",        "nunique"),
            first_purchase  = ("invoice_date",  "min"),
        )
        .reset_index()
    )

    rfm["recency_days"] = (snapshot_date - rfm["last_purchase"]).dt.days

    # ── Derived features ────────────────────────────────────
    rfm["avg_order_value"]    = (rfm["monetary"] / rfm["frequency"]).round(2)
    rfm["avg_items_per_order"]= (rfm["total_items"] / rfm["frequency"]).round(2)
    rfm["active_days"]        = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days + 1
    rfm["revenue_per_day"]    = (rfm["monetary"] / rfm["active_days"]).round(4)

    # Customer's most frequent country
    top_country = (
        df.groupby(["customer_id", "country"])["invoice_no"]
        .nunique()
        .reset_index()
        .sort_values("invoice_no", ascending=False)
        .drop_duplicates("customer_id")[["customer_id", "country"]]
        .rename(columns={"country": "top_country"})
    )
    rfm = rfm.merge(top_country, on="customer_id", how="left")

    # ── RFM quintile scores (1–5) ──────────────────────────
    # Recency: lower days = better = score 5
    rfm["r_score"] = pd.qcut(
        rfm["recency_days"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop"
    ).astype(int)
    rfm["f_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["m_score"] = pd.qcut(
        rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["rfm_total"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    # ── Rule-based RFM label (for reference / EDA) ─────────
    rfm["rfm_label"] = rfm["rfm_total"].apply(_rfm_label)

    # ── Drop helper columns not needed downstream ──────────
    rfm = rfm.drop(columns=["last_purchase", "first_purchase"])

    print(f"\n  Customer feature matrix: {rfm.shape[0]:,} rows x {rfm.shape[1]} cols")
    return rfm


def _rfm_label(score: int) -> str:
    if score >= 13:
        return "Champions"
    if score >= 10:
        return "Loyal Customers"
    if score >= 7:
        return "At Risk"
    return "Needs Attention"


# ─────────────────────────────────────────────────────────────
# Feature columns used in clustering
# ─────────────────────────────────────────────────────────────

CLUSTER_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "avg_items_per_order",
    "unique_products",
]

# Features used by the WTP model
WTP_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "avg_items_per_order",
    "unique_products",
    "revenue_per_day",
    "active_days",
]


# ─────────────────────────────────────────────────────────────
# Scaling helper
# ─────────────────────────────────────────────────────────────

from sklearn.preprocessing import StandardScaler


def scale_features(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, StandardScaler]:
    """
    Standard-scale selected columns. Returns (X_scaled, fitted_scaler).
    Keep the scaler to inverse-transform later.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols].fillna(0))
    return X, scaler


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.clean_data import clean

    print("Running cleaning pipeline first...")
    df_clean = clean()
    print("\nBuilding customer features...")
    customers = build_customer_features(df_clean)
    print(customers.describe().round(2).to_string())
    customers.to_csv("data/customers.csv", index=False)
    print("\nSaved to data/customers.csv")
