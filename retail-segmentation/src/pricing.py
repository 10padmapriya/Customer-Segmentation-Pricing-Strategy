"""
src/pricing.py
==============
Translates segment profiles and WTP predictions into a concrete
pricing strategy with tier definitions and MRR uplift projections.

The four tiers map directly to the four GMM segments:
  Champions       → Enterprise tier  ($349/mo)
  Loyal Customers → Pro tier         ($79/mo)
  At Risk         → Win-Back bundle  ($47/mo, time-limited)
  Needs Attention → Starter/Free     ($0 → $19/mo upgrade path)

Tier prices are anchored to:
  - The WTP prediction percentiles per segment
  - A 20% discount to the median WTP (to maximise conversion)

Usage:
    from src.pricing import build_pricing_strategy, print_strategy
    strategy = build_pricing_strategy(customers)
"""

from __future__ import annotations
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# Segment → tier mapping
# ─────────────────────────────────────────────────────────────

TIER_MAP = {
    "Champions":       "Enterprise",
    "Loyal Customers": "Pro",
    "At Risk":         "Win-Back",
    "Needs Attention": "Starter",
}

TIER_PRICES = {
    "Enterprise": 349,
    "Pro":         79,
    "Win-Back":    47,    # 40% off Pro, time-limited 3 months
    "Starter":      0,    # Freemium — upgrade path to $19
}

TIER_COLORS = {
    "Enterprise": "#4f8ef7",
    "Pro":        "#2dd4bf",
    "Win-Back":   "#f97316",
    "Starter":    "#f59e0b",
}


# ─────────────────────────────────────────────────────────────
# Core builder
# ─────────────────────────────────────────────────────────────

def build_pricing_strategy(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Build a segment-level pricing summary from the customer DataFrame.

    Requires columns: segment_name, monetary, wtp_predicted, churn_risk.

    Returns:
        DataFrame with one row per segment containing:
          segment_name, n_customers, pct_of_base,
          avg_monetary, median_wtp, p25_wtp, p75_wtp,
          recommended_tier, recommended_price,
          est_conversion_pct, est_mrr_uplift
    """
    rows = []
    total = len(customers)

    for seg_name, group in customers.groupby("segment_name"):
        tier  = TIER_MAP.get(seg_name, "Starter")
        price = TIER_PRICES[tier]

        # WTP stats
        wtp_col = "wtp_predicted" if "wtp_predicted" in group.columns else "wtp_proxy"
        median_wtp = group[wtp_col].median() if wtp_col in group.columns else 0
        p25_wtp    = group[wtp_col].quantile(0.25) if wtp_col in group.columns else 0
        p75_wtp    = group[wtp_col].quantile(0.75) if wtp_col in group.columns else 0

        # Conversion probability: customers whose WTP ≥ tier price
        if wtp_col in group.columns and price > 0:
            conversion = (group[wtp_col] >= price).mean()
        elif price == 0:
            conversion = 0.85   # Freemium has high "adoption" rate
        else:
            conversion = 0.30

        n_converting     = int(len(group) * conversion)
        est_mrr_uplift   = n_converting * price

        rows.append({
            "segment_name":        seg_name,
            "n_customers":         len(group),
            "pct_of_base":         round(len(group) / total * 100, 1),
            "avg_monetary":        round(group["monetary"].mean(), 2),
            "avg_order_value":     round(group["avg_order_value"].mean(), 2),
            "avg_churn_risk":      round(group["churn_risk"].mean(), 3),
            "median_wtp":          round(median_wtp, 2),
            "p25_wtp":             round(p25_wtp, 2),
            "p75_wtp":             round(p75_wtp, 2),
            "recommended_tier":    tier,
            "recommended_price":   price,
            "est_conversion_pct":  round(conversion * 100, 1),
            "est_converting":      n_converting,
            "est_mrr_uplift":      est_mrr_uplift,
        })

    result = pd.DataFrame(rows).sort_values("avg_monetary", ascending=False)
    return result


def print_strategy(strategy: pd.DataFrame) -> None:
    """Pretty-print the pricing strategy to stdout."""
    divider = "─" * 80
    print(f"\n{divider}")
    print("  PRICING STRATEGY RECOMMENDATIONS")
    print(divider)
    print(f"  {'Segment':<22} {'Tier':<12} {'Price':>6}  "
          f"{'Customers':>10}  {'Conv%':>6}  {'MRR Uplift':>12}")
    print(divider)

    total_mrr = 0
    for _, row in strategy.iterrows():
        print(f"  {row['segment_name']:<22} {row['recommended_tier']:<12} "
              f"${row['recommended_price']:>5}  "
              f"{row['n_customers']:>10,}  "
              f"{row['est_conversion_pct']:>5.1f}%  "
              f"£{row['est_mrr_uplift']:>11,}")
        total_mrr += row["est_mrr_uplift"]

    print(divider)
    print(f"  {'Total estimated MRR uplift':>56}  £{total_mrr:>11,}")
    print(divider)


def bundle_wtp_increments() -> pd.DataFrame:
    """
    Feature-level WTP increments from conjoint analysis logic.
    Used for the bundle visualisation.
    These are estimated — in a real project you'd run a conjoint survey.
    """
    return pd.DataFrame([
        {"feature": "Core platform",       "incremental_wtp": 79},
        {"feature": "Analytics suite",     "incremental_wtp": 49},
        {"feature": "API access",          "incremental_wtp": 45},
        {"feature": "Collaboration seats", "incremental_wtp": 30},
        {"feature": "Priority support",    "incremental_wtp": 21},
        {"feature": "Export tools",        "incremental_wtp": 18},
    ])
