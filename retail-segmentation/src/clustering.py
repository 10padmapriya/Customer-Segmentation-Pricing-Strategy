"""
src/clustering.py
=================
Three clustering algorithms applied to the customer feature matrix.

  KMeans  — fast baseline, spherical clusters
  DBSCAN  — density-based, finds outliers automatically
  GMM     — probabilistic soft assignments (selected model)

Each function returns a result dict so you can compare them cleanly.
GMM is selected because:
  - Soft probabilities feed into the WTP model as features
  - Handles the elliptical cluster shapes in RFM space better than KMeans
  - Best BIC score on this dataset

Usage:
    from src.clustering import run_all_models, label_segments
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# ─────────────────────────────────────────────────────────────
# KMeans
# ─────────────────────────────────────────────────────────────

def run_kmeans(
    X: np.ndarray,
    k: int = 4,
    random_state: int = 42,
    n_init: int = 20,
) -> dict:
    """
    Fit KMeans and return labels + evaluation metrics.

    Args:
        X            : Scaled feature matrix (n_samples, n_features).
        k            : Number of clusters.
        random_state : Seed for reproducibility.
        n_init       : Number of random initialisations (higher = more stable).

    Returns:
        dict with: labels, model, silhouette, davies_bouldin,
                   calinski_harabasz, inertia.
    """
    model  = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)

    result = {
        "model":              model,
        "labels":             labels,
        "silhouette":         round(silhouette_score(X, labels), 4),
        "davies_bouldin":     round(davies_bouldin_score(X, labels), 4),
        "calinski_harabasz":  round(calinski_harabasz_score(X, labels), 2),
        "inertia":            round(model.inertia_, 2),
    }
    print(f"  KMeans  k={k} | sil={result['silhouette']:.3f} "
          f"| DB={result['davies_bouldin']:.3f} "
          f"| inertia={result['inertia']:,.0f}")
    return result


def elbow_curve(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run KMeans for k in k_range. Returns a DataFrame for plotting.

    Columns: k, inertia, silhouette.
    Use the elbow in inertia and the peak in silhouette to pick k.
    """
    rows = []
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        rows.append({
            "k":          k,
            "inertia":    round(km.inertia_, 2),
            "silhouette": round(silhouette_score(X, labels), 4),
        })
        print(f"    k={k} inertia={rows[-1]['inertia']:>12,.0f}  sil={rows[-1]['silhouette']:.4f}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# DBSCAN
# ─────────────────────────────────────────────────────────────

def run_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 10,
) -> dict:
    """
    Fit DBSCAN and return labels + evaluation metrics.

    Noise points (label == -1) are excluded from silhouette calculation.
    They represent customers that don't fit any segment — which is useful
    information on its own (outliers, big-spending one-offs, etc.).

    Args:
        X          : Scaled feature matrix.
        eps        : Max distance between two samples to be in the same neighbourhood.
        min_samples: Min samples in a neighbourhood to form a core point.

    Returns:
        dict with: labels, model, silhouette, n_clusters, noise_count, noise_pct.
    """
    model  = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    valid      = labels != -1
    n_clusters = len(set(labels[valid]))
    noise_count = (~valid).sum()
    noise_pct   = round((~valid).mean() * 100, 2)

    sil = (
        round(silhouette_score(X[valid], labels[valid]), 4)
        if valid.sum() > 1 else 0.0
    )

    result = {
        "model":       model,
        "labels":      labels,
        "silhouette":  sil,
        "n_clusters":  n_clusters,
        "noise_count": int(noise_count),
        "noise_pct":   noise_pct,
    }
    print(f"  DBSCAN  eps={eps} | clusters={n_clusters} "
          f"| noise={noise_pct:.1f}% ({noise_count:,} customers) "
          f"| sil={sil:.3f}")
    return result


# ─────────────────────────────────────────────────────────────
# GMM  (selected model)
# ─────────────────────────────────────────────────────────────

def run_gmm(
    X: np.ndarray,
    n_components: int = 4,
    covariance_type: str = "full",
    random_state: int = 42,
    max_iter: int = 300,
) -> dict:
    """
    Fit a Gaussian Mixture Model and return labels, soft probabilities,
    and evaluation metrics.

    Why GMM?
      • Soft probabilities — a customer can be 70% Champion / 30% At-Risk.
        These probabilities become features in the WTP model.
      • Full covariance captures elliptical clusters (RFM data is skewed).
      • BIC/AIC allow principled model selection across different n_components.

    Args:
        X               : Scaled feature matrix.
        n_components    : Number of Gaussian components (= segments).
        covariance_type : 'full' allows each component its own covariance matrix.
        random_state    : Seed.
        max_iter        : EM algorithm iteration limit.

    Returns:
        dict with: labels, proba, model, silhouette, bic, aic, log_likelihood.
    """
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=max_iter,
        n_init=5,
    )
    model.fit(X)

    labels = model.predict(X)
    proba  = model.predict_proba(X)

    result = {
        "model":          model,
        "labels":         labels,
        "proba":          proba,
        "silhouette":     round(silhouette_score(X, labels), 4),
        "bic":            round(model.bic(X), 2),
        "aic":            round(model.aic(X), 2),
        "log_likelihood": round(model.score(X) * len(X), 2),
    }
    print(f"  GMM     n={n_components} | sil={result['silhouette']:.3f} "
          f"| BIC={result['bic']:,.0f} | AIC={result['aic']:,.0f}  ← SELECTED")
    return result


def gmm_bic_search(
    X: np.ndarray,
    n_range: range = range(2, 8),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit GMM for each n_components and return BIC/AIC/silhouette.
    Lower BIC = better model. Use to pick n_components.
    """
    rows = []
    for n in n_range:
        gmm    = GaussianMixture(n_components=n, covariance_type="full",
                                 random_state=random_state, n_init=5)
        gmm.fit(X)
        labels = gmm.predict(X)
        rows.append({
            "n_components": n,
            "bic":          round(gmm.bic(X), 2),
            "aic":          round(gmm.aic(X), 2),
            "silhouette":   round(silhouette_score(X, labels), 4),
        })
        print(f"    n={n}  BIC={rows[-1]['bic']:>12,.0f}  "
              f"AIC={rows[-1]['aic']:>12,.0f}  sil={rows[-1]['silhouette']:.4f}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Run all three and compare
# ─────────────────────────────────────────────────────────────

def run_all_models(X: np.ndarray, k: int = 4) -> dict[str, dict]:
    """
    Convenience function — fits KMeans, DBSCAN, and GMM in one call.

    Returns:
        {'kmeans': {...}, 'dbscan': {...}, 'gmm': {...}}
    """
    print("\nFitting clustering models...")
    km = run_kmeans(X, k=k)
    db = run_dbscan(X)
    gm = run_gmm(X, n_components=k)
    return {"kmeans": km, "dbscan": db, "gmm": gm}


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a side-by-side comparison table of all three models.

    Args:
        results: Output of run_all_models().

    Returns:
        pd.DataFrame with metrics as rows and models as columns.
    """
    km, db, gm = results["kmeans"], results["dbscan"], results["gmm"]
    data = {
        "KMeans":  [km["silhouette"], km["davies_bouldin"], km["inertia"],       "-",        "-"],
        "DBSCAN":  [db["silhouette"], "-",                  "-",                 db["n_clusters"], db["noise_pct"]],
        "GMM ✓":   [gm["silhouette"], "-",                  "-",                 "-",        "-"],
    }
    index = ["Silhouette ↑", "Davies-Bouldin ↓", "Inertia ↓", "Clusters found", "Noise %"]
    return pd.DataFrame(data, index=index)


# ─────────────────────────────────────────────────────────────
# Segment labelling (maps cluster IDs to business names)
# ─────────────────────────────────────────────────────────────

def label_segments(
    customers: pd.DataFrame,
    labels: np.ndarray,
    proba: np.ndarray,
) -> pd.DataFrame:
    """
    Attach GMM cluster labels and soft probabilities to the customer DataFrame.
    Also auto-derives a human-readable segment name from RFM means.

    The segment name mapping is done by ranking clusters on RFM and assigning:
      Highest monetary+frequency, lowest recency → "Champions"
      High RFM but not top                       → "Loyal Customers"
      Low recency (not purchased recently)       → "At Risk"
      Low across all                             → "Casual / Low-Value"

    Args:
        customers : Customer feature DataFrame (from src/features.py).
        labels    : GMM hard labels (n_customers,).
        proba     : GMM soft probabilities (n_customers, n_segments).

    Returns:
        customers DataFrame with new columns:
          segment_id, gmm_p0..gmm_p3, segment_name.
    """
    out = customers.copy()
    out["segment_id"] = labels

    # Attach soft probabilities
    for i in range(proba.shape[1]):
        out[f"gmm_p{i}"] = proba[:, i].round(4)

    # Derive segment names from cluster means
    seg_means = out.groupby("segment_id")[["recency_days", "frequency", "monetary"]].mean()

    # Rank: low recency = good (recently active), high freq+monetary = good
    seg_means["recency_rank"]  = seg_means["recency_days"].rank()          # low = best
    seg_means["frequency_rank"]= seg_means["frequency"].rank(ascending=False)
    seg_means["monetary_rank"] = seg_means["monetary"].rank(ascending=False)
    seg_means["overall_rank"]  = (
        seg_means["recency_rank"] +
        seg_means["frequency_rank"] +
        seg_means["monetary_rank"]
    )
    seg_means = seg_means.sort_values("overall_rank")

    n = len(seg_means)
    name_map = {}
    names = ["Champions", "Loyal Customers", "At Risk", "Needs Attention"]
    # Pad if more than 4 segments
    while len(names) < n:
        names.append(f"Segment {len(names)}")

    for rank_pos, seg_id in enumerate(seg_means.index):
        name_map[seg_id] = names[rank_pos]

    out["segment_name"] = out["segment_id"].map(name_map)

    print("\n  Segment sizes:")
    for name, count in out["segment_name"].value_counts().items():
        pct = count / len(out) * 100
        print(f"    {name:<22} {count:>5,}  ({pct:.1f}%)")

    return out
