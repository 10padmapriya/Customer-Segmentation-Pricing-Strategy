"""
main.py
=======
Single entry point — runs the complete pipeline end-to-end.

Stages:
  1. Load raw data          (downloads from UCI if not cached)
  2. Clean data             (nulls, cancellations, bad codes)
  3. Feature engineering    (RFM + behavioral signals)
  4. Elbow + BIC analysis   (find optimal k)
  5. Clustering             (KMeans · DBSCAN · GMM)
  6. Segment labelling      (Champions / Loyal / At Risk / Needs Attention)
  7. Churn risk scoring
  8. WTP modelling + SHAP
  9. Pricing strategy
  10. Visualisations

Usage:
    python main.py
    python main.py --skip-download    (if raw_retail.csv already exists)
    python main.py --no-plots         (skip PNG generation — faster)
    python main.py --k 5              (override cluster count)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

DIVIDER = "=" * 65


def step(n: int, title: str) -> None:
    print(f"\n[{n}] {title}")
    print("    " + "─" * (len(title) + 4))


def main(skip_download: bool = False, plots: bool = True, k: int = 4) -> dict:
    print("\n" + DIVIDER)
    print("  CUSTOMER SEGMENTATION + PRICING STRATEGY")
    print("  UCI Online Retail II  —  Real Transactions 2009-2011")
    print(DIVIDER)

    # ── 1. Raw data ───────────────────────────────────────────
    step(1, "Loading raw data")
    raw_path = Path("data/raw_retail.csv")

    if not raw_path.exists() or not skip_download:
        from data.load_data import download_raw_data
        df_raw = download_raw_data()
    else:
        print(f"  Using cached {raw_path}")
        df_raw = pd.read_csv(raw_path, low_memory=False)

    print(f"  Raw rows: {len(df_raw):,}")

    # ── 2. Cleaning ───────────────────────────────────────────
    step(2, "Cleaning data")
    from data.clean_data import clean
    df_clean = clean(raw_path="data/raw_retail.csv")

    # ── 3. Feature engineering ────────────────────────────────
    step(3, "Building customer feature matrix")
    from src.features import build_customer_features, scale_features, CLUSTER_FEATURES
    customers = build_customer_features(df_clean)

    # ── 4. Elbow + BIC to find optimal k ─────────────────────
    step(4, "Model selection — elbow curve + GMM BIC")
    from src.clustering import elbow_curve, gmm_bic_search

    X, scaler = scale_features(customers, CLUSTER_FEATURES)

    print("\n  KMeans elbow:")
    elbow_df = elbow_curve(X, k_range=range(2, 9))

    print("\n  GMM BIC search:")
    bic_df = gmm_bic_search(X, n_range=range(2, 7))

    # ── 5. Clustering ─────────────────────────────────────────
    step(5, f"Running all three clustering models (k={k})")
    from src.clustering import run_all_models, comparison_table

    results = run_all_models(X, k=k)

    print("\n  Model comparison:")
    cmp = comparison_table(results)
    print(cmp.to_string())

    # ── 6. Label segments ─────────────────────────────────────
    step(6, "Labelling segments (GMM selected)")
    from src.clustering import label_segments

    gmm_result = results["gmm"]
    customers  = label_segments(customers, gmm_result["labels"], gmm_result["proba"])

    # Build segment name → id map for PCA plot
    seg_name_map = dict(zip(customers["segment_id"], customers["segment_name"]))

    # ── 7. Churn risk ─────────────────────────────────────────
    step(7, "Churn risk scoring")
    from src.churn import compute_churn_risk, flag_high_risk, churn_summary

    customers["churn_risk"] = compute_churn_risk(customers)
    high_risk = flag_high_risk(customers)

    print("\n  Churn summary by segment:")
    print(churn_summary(customers).to_string(index=False))

    # ── 8. WTP model + SHAP ───────────────────────────────────
    step(8, "WTP model — Gradient Boosting + SHAP")
    from src.wtp_model import (
        build_wtp_target, train_wtp_model,
        predict_wtp, compute_shap, print_shap_bar,
    )

    customers["wtp_proxy"] = build_wtp_target(customers)
    wtp_model, wtp_features, metrics = train_wtp_model(customers)
    customers["wtp_predicted"] = predict_wtp(wtp_model, customers, wtp_features)

    shap_vals, X_sample = compute_shap(wtp_model, customers, wtp_features)
    print_shap_bar(shap_vals, X_sample)

    # ── 9. Pricing strategy ───────────────────────────────────
    step(9, "Pricing strategy")
    from src.pricing import build_pricing_strategy, print_strategy

    strategy = build_pricing_strategy(customers)
    print_strategy(strategy)

    # ── 10. Visualisations ────────────────────────────────────
    if plots:
        step(10, "Generating visualisations → outputs/")
        from src.visualize import (
            plot_rfm_scatter, plot_elbow_curve, plot_gmm_bic,
            plot_cluster_pca, plot_churn_distribution,
            plot_segment_revenue, plot_wtp_distribution,
            plot_shap_summary, plot_pricing_strategy,
        )
        plot_rfm_scatter(customers)
        plot_elbow_curve(elbow_df)
        plot_gmm_bic(bic_df)
        plot_cluster_pca(X, gmm_result["labels"], "GMM Clusters", seg_name_map)
        plot_cluster_pca(X, results["kmeans"]["labels"], "KMeans Clusters")
        plot_churn_distribution(customers)
        plot_segment_revenue(customers)
        plot_wtp_distribution(customers)
        plot_shap_summary(shap_vals, X_sample)
        plot_pricing_strategy(strategy)
    else:
        print("\n[10] Plots skipped (--no-plots)")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + DIVIDER)
    print("  PIPELINE COMPLETE")
    print(f"  Raw transactions   : {len(df_raw):,}")
    print(f"  Cleaned rows       : {len(df_clean):,}")
    print(f"  Customers          : {len(customers):,}")
    print(f"  Segments           : {customers['segment_name'].nunique()}")
    print(f"  GMM silhouette     : {gmm_result['silhouette']:.3f}")
    print(f"  WTP model R²       : {metrics['r2_test']:.3f}")
    print(f"  High-risk churn    : {len(high_risk):,} ({len(high_risk)/len(customers):.1%})")
    print(f"  Est. MRR uplift    : £{strategy['est_mrr_uplift'].sum():,}/mo")
    if plots:
        print(f"  Plots saved to     : outputs/")
    print(DIVIDER + "\n")

    return {
        "customers": customers,
        "df_clean":  df_clean,
        "elbow_df":  elbow_df,
        "bic_df":    bic_df,
        "results":   results,
        "strategy":  strategy,
        "wtp_model": wtp_model,
        "shap_vals": shap_vals,
        "X_sample":  X_sample,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Customer Segmentation + Pricing Pipeline"
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Use cached raw_retail.csv (skip UCI download)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--k", type=int, default=4,
                        help="Number of clusters (default: 4)")
    args = parser.parse_args()

    main(
        skip_download=args.skip_download,
        plots=not args.no_plots,
        k=args.k,
    )
