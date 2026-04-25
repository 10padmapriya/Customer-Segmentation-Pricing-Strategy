"""
src/visualize.py
================
All project visualisations. Every function saves a PNG to outputs/.

Plots:
  1.  plot_rfm_scatter          — customers in RFM space, coloured by segment
  2.  plot_elbow_curve          — KMeans inertia + silhouette vs k
  3.  plot_gmm_bic              — GMM BIC/AIC vs n_components
  4.  plot_cluster_pca          — 2D PCA projection of clusters
  5.  plot_churn_distribution   — churn risk histograms by segment
  6.  plot_segment_revenue      — total spend per segment
  7.  plot_wtp_distribution     — WTP distribution per segment
  8.  plot_shap_summary         — SHAP feature importance
  9.  plot_pricing_strategy     — MRR uplift per tier

Usage:
    from src.visualize import plot_rfm_scatter
    plot_rfm_scatter(customers)
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap


# ─────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

DARK_BG = "#0b0c0f"
SURFACE = "#13151a"
MUTED   = "#7b7f91"
TEXT    = "#e8eaf0"
BORDER  = "#2a2d37"

SEG_COLORS = {
    "Champions":       "#4f8ef7",
    "Loyal Customers": "#2dd4bf",
    "At Risk":         "#f97316",
    "Needs Attention": "#f59e0b",
}
DEFAULT_COLORS = ["#4f8ef7", "#2dd4bf", "#f97316", "#f59e0b",
                  "#a78bfa", "#ec4899", "#34d399", "#fb923c"]

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   MUTED,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        "#1c1f27",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


def _save(name: str) -> None:
    path = OUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  Saved → {path}")
    plt.close()


def _seg_color(seg: str) -> str:
    return SEG_COLORS.get(seg, "#888888")


def _legend_patches(segments: list[str]) -> list[mpatches.Patch]:
    return [mpatches.Patch(color=_seg_color(s), label=s) for s in segments]


# ─────────────────────────────────────────────────────────────
# 1. RFM Scatter
# ─────────────────────────────────────────────────────────────

def plot_rfm_scatter(customers: pd.DataFrame, seg_col: str = "segment_name") -> None:
    """
    Three scatter plots showing customers in RFM space, coloured by segment.
    Rasterized for performance with large datasets.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Customer Distribution in RFM Space", fontsize=14,
                 fontweight="bold", color=TEXT, y=1.01)

    pairs = [
        ("recency_days",  "monetary",   "Recency (days)", "Monetary (£)"),
        ("frequency",     "monetary",   "Frequency (orders)", "Monetary (£)"),
        ("recency_days",  "frequency",  "Recency (days)", "Frequency (orders)"),
    ]
    segments = customers[seg_col].unique()

    for ax, (xc, yc, xl, yl) in zip(axes, pairs):
        for seg in segments:
            mask = customers[seg_col] == seg
            ax.scatter(
                customers.loc[mask, xc],
                customers.loc[mask, yc],
                alpha=0.25, s=6,
                color=_seg_color(seg),
                label=seg, rasterized=True,
            )
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.grid(True)

    segs = [s for s in SEG_COLORS if s in segments]
    fig.legend(handles=_legend_patches(segs), loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05), fontsize=10)
    plt.tight_layout()
    _save("01_rfm_scatter.png")


# ─────────────────────────────────────────────────────────────
# 2. Elbow Curve
# ─────────────────────────────────────────────────────────────

def plot_elbow_curve(elbow_df: pd.DataFrame) -> None:
    """
    Plot KMeans inertia (left axis) and silhouette score (right axis) vs k.
    The elbow in inertia + peak in silhouette jointly suggest optimal k.
    """
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(elbow_df["k"], elbow_df["inertia"],    "o-",  color="#4f8ef7",
             lw=2, label="Inertia", zorder=3)
    ax2.plot(elbow_df["k"], elbow_df["silhouette"], "s--", color="#2dd4bf",
             lw=2, label="Silhouette", zorder=3)
    ax1.axvline(4, color="#f59e0b", linestyle=":", lw=1.5, label="k=4 selected")

    ax1.set_xlabel("k (clusters)"); ax1.set_ylabel("Inertia", color="#4f8ef7")
    ax2.set_ylabel("Silhouette Score", color="#2dd4bf")
    ax1.tick_params(axis="y", labelcolor="#4f8ef7")
    ax2.tick_params(axis="y", labelcolor="#2dd4bf")
    ax2.set_facecolor(SURFACE)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=10)
    fig.suptitle("KMeans — Elbow Curve & Silhouette Score", fontsize=13,
                 fontweight="bold", color=TEXT)
    plt.tight_layout()
    _save("02_elbow_curve.png")


# ─────────────────────────────────────────────────────────────
# 3. GMM BIC / AIC
# ─────────────────────────────────────────────────────────────

def plot_gmm_bic(bic_df: pd.DataFrame) -> None:
    """
    Plot GMM BIC and AIC vs n_components.
    Lower BIC/AIC = better model fit. Pick the elbow.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(bic_df["n_components"], bic_df["bic"], "o-", color="#4f8ef7",
            lw=2, label="BIC")
    ax.plot(bic_df["n_components"], bic_df["aic"], "s--", color="#2dd4bf",
            lw=2, label="AIC")

    best_n = bic_df.loc[bic_df["bic"].idxmin(), "n_components"]
    ax.axvline(best_n, color="#f59e0b", linestyle=":", lw=1.5,
               label=f"n={best_n} selected")

    ax.set_xlabel("n_components"); ax.set_ylabel("Score (lower = better)")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True)
    fig.suptitle("GMM — BIC / AIC Model Selection", fontsize=13,
                 fontweight="bold", color=TEXT)
    plt.tight_layout()
    _save("03_gmm_bic.png")


# ─────────────────────────────────────────────────────────────
# 4. PCA cluster projection
# ─────────────────────────────────────────────────────────────

def plot_cluster_pca(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    title: str = "GMM Clusters — PCA Projection",
    seg_names: dict[int, str] | None = None,
) -> None:
    """
    Project the scaled feature matrix to 2D using PCA and colour by cluster.

    Args:
        X_scaled  : Scaled feature matrix (n_customers, n_features).
        labels    : Cluster label per customer.
        title     : Plot title and filename suffix.
        seg_names : Optional dict mapping cluster int → segment name string.
    """
    from sklearn.decomposition import PCA
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var    = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 7))
    unique_labels = sorted(set(labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = seg_names.get(label, f"Cluster {label}") if seg_names else f"Cluster {label}"
        color = (DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                 if label != -1 else "#555555")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=5, alpha=0.30, color=color,
                   label=f"{name} (n={mask.sum():,})", rasterized=True)

    ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)")
    ax.legend(frameon=False, fontsize=9, markerscale=3)
    ax.grid(True)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=TEXT)
    plt.tight_layout()
    fname = "04_" + title.lower().replace(" ", "_").replace("—", "").replace("/", "") + ".png"
    _save(fname)


# ─────────────────────────────────────────────────────────────
# 5. Churn distribution
# ─────────────────────────────────────────────────────────────

def plot_churn_distribution(
    customers: pd.DataFrame,
    risk_col: str = "churn_risk",
    seg_col: str  = "segment_name",
) -> None:
    """
    Histogram + boxplot of churn risk by segment.
    Segments with high risk show right-skewed histograms.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Churn Risk Distribution by Segment", fontsize=13,
                 fontweight="bold", color=TEXT)

    # Histogram
    for seg, grp in customers.groupby(seg_col):
        axes[0].hist(grp[risk_col], bins=30, alpha=0.5,
                     color=_seg_color(seg), label=seg, density=True)
    axes[0].axvline(0.65, color="#ef4444", lw=1.5, linestyle="--", label="High-risk threshold")
    axes[0].set_xlabel("Churn Risk Score")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(True)

    # Boxplot
    order = [s for s in SEG_COLORS if s in customers[seg_col].unique()]
    data  = [customers.loc[customers[seg_col] == s, risk_col].values for s in order]
    bp    = axes[1].boxplot(
        data,
        patch_artist=True,
        labels=[s.replace(" ", "\n") for s in order],
        medianprops=dict(color="#ffffff", lw=2),
        whiskerprops=dict(color=MUTED),
        capprops=dict(color=MUTED),
        flierprops=dict(marker=".", markersize=2, color=MUTED, alpha=0.3),
    )
    for patch, seg in zip(bp["boxes"], order):
        patch.set_facecolor(_seg_color(seg))
        patch.set_alpha(0.75)
    axes[1].set_ylabel("Churn Risk Score")
    axes[1].grid(True, axis="y")

    plt.tight_layout()
    _save("05_churn_distribution.png")


# ─────────────────────────────────────────────────────────────
# 6. Segment revenue
# ─────────────────────────────────────────────────────────────

def plot_segment_revenue(
    customers: pd.DataFrame,
    monetary_col: str = "monetary",
    seg_col: str      = "segment_name",
) -> None:
    """Bar chart: total spend per segment + share of total revenue."""
    order   = [s for s in SEG_COLORS if s in customers[seg_col].unique()]
    summary = (
        customers.groupby(seg_col)[monetary_col]
        .agg(["sum", "mean", "count"])
        .reindex(order)
        .reset_index()
    )
    total_rev = summary["sum"].sum()
    summary["pct"] = summary["sum"] / total_rev * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        summary[seg_col],
        summary["sum"] / 1_000,
        color=[_seg_color(s) for s in summary[seg_col]],
        alpha=0.85, width=0.55, edgecolor="none",
    )
    for bar, row in zip(bars, summary.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"£{bar.get_height():,.0f}K\n({row.pct:.1f}%)",
            ha="center", fontsize=10, fontweight="bold", color=TEXT,
        )
    ax.set_ylabel("Total Revenue (£K)")
    ax.set_xlabel("")
    ax.set_title("Revenue by Segment", fontsize=13, fontweight="bold", color=TEXT)
    ax.grid(axis="y")
    plt.tight_layout()
    _save("06_segment_revenue.png")


# ─────────────────────────────────────────────────────────────
# 7. WTP distribution
# ─────────────────────────────────────────────────────────────

def plot_wtp_distribution(
    customers: pd.DataFrame,
    wtp_col: str = "wtp_proxy",
    seg_col: str = "segment_name",
) -> None:
    """2×2 grid of WTP histograms, one per segment."""
    segs   = [s for s in SEG_COLORS if s in customers[seg_col].unique()]
    n_segs = len(segs)
    cols   = 2
    rows   = (n_segs + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten()
    fig.suptitle("WTP Distribution by Segment", fontsize=13,
                 fontweight="bold", color=TEXT)

    for i, seg in enumerate(segs):
        grp    = customers.loc[customers[seg_col] == seg, wtp_col].dropna()
        color  = _seg_color(seg)
        axes[i].hist(grp, bins=40, color=color, alpha=0.85, edgecolor="none")
        median = grp.median()
        axes[i].axvline(median, color="#ffffff", lw=1.5, linestyle="--",
                        label=f"Median: £{median:.1f}")
        axes[i].set_title(seg, fontsize=11, color=color)
        axes[i].set_xlabel("WTP Score"); axes[i].set_ylabel("Customers")
        axes[i].legend(frameon=False, fontsize=9)
        axes[i].grid(True)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save("07_wtp_distribution.png")


# ─────────────────────────────────────────────────────────────
# 8. SHAP summary
# ─────────────────────────────────────────────────────────────

def plot_shap_summary(shap_vals: np.ndarray, X_sample: pd.DataFrame) -> None:
    """
    SHAP beeswarm plot — shows both importance (x-axis range) and
    direction (red = high feature value pushes WTP up).
    """
    fig = plt.figure(figsize=(10, 7), facecolor=DARK_BG)
    shap.summary_plot(shap_vals, X_sample, show=False, plot_size=(10, 7))
    plt.title("SHAP — WTP Model Feature Importance",
              fontsize=13, fontweight="bold", color=TEXT)
    plt.tight_layout()
    _save("08_shap_summary.png")


# ─────────────────────────────────────────────────────────────
# 9. Pricing strategy
# ─────────────────────────────────────────────────────────────

def plot_pricing_strategy(strategy: pd.DataFrame) -> None:
    """
    Horizontal bar chart of estimated MRR uplift per tier,
    with conversion rate annotated on each bar.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    colors = {
        "Enterprise": "#4f8ef7",
        "Pro":        "#2dd4bf",
        "Win-Back":   "#f97316",
        "Starter":    "#f59e0b",
    }
    tier_col  = strategy["recommended_tier"]
    mrr_col   = strategy["est_mrr_uplift"]
    seg_col   = strategy["segment_name"]

    bar_colors = [colors.get(t, "#888") for t in tier_col]
    bars = ax.barh(seg_col, mrr_col, color=bar_colors, alpha=0.85, edgecolor="none")

    for bar, row in zip(bars, strategy.itertuples()):
        ax.text(
            bar.get_width() + mrr_col.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"£{row.est_mrr_uplift:,}  ({row.est_conversion_pct:.0f}% conv.)",
            va="center", fontsize=10, color=TEXT,
        )
    ax.set_xlabel("Estimated MRR Uplift (£)")
    ax.set_title("Pricing Strategy — Projected MRR Uplift by Segment",
                 fontsize=13, fontweight="bold", color=TEXT)
    ax.grid(axis="x")
    ax.set_xlim(right=mrr_col.max() * 1.45)
    plt.tight_layout()
    _save("09_pricing_strategy.png")
