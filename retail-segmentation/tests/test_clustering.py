"""
Tests for src/clustering.py
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering import (
    run_kmeans, run_dbscan, run_gmm,
    elbow_curve, gmm_bic_search,
    run_all_models, comparison_table, label_segments,
)


@pytest.fixture
def X_blobs():
    """Four well-separated Gaussian blobs — easy for all algorithms."""
    np.random.seed(42)
    blobs = [
        np.random.normal([5, 5, 5],    1, (80, 3)),
        np.random.normal([20, 20, 20], 1, (80, 3)),
        np.random.normal([5, 20, 5],   1, (80, 3)),
        np.random.normal([20, 5, 20],  1, (80, 3)),
    ]
    return np.vstack(blobs)


@pytest.fixture
def customers_df():
    """Minimal customer DataFrame for label_segments."""
    np.random.seed(0)
    n = 320
    return pd.DataFrame({
        "customer_id":         [f"C{i}" for i in range(n)],
        "recency_days":        np.random.randint(1, 100, n),
        "frequency":           np.random.randint(1, 30, n),
        "monetary":            np.random.uniform(10, 2000, n),
        "avg_order_value":     np.random.uniform(5, 200, n),
        "avg_items_per_order": np.random.uniform(1, 20, n),
        "unique_products":     np.random.randint(1, 50, n),
    })


class TestKMeans:
    def test_returns_all_keys(self, X_blobs):
        r = run_kmeans(X_blobs, k=4)
        for key in ["model", "labels", "silhouette", "davies_bouldin",
                    "calinski_harabasz", "inertia"]:
            assert key in r

    def test_label_length(self, X_blobs):
        r = run_kmeans(X_blobs, k=4)
        assert len(r["labels"]) == len(X_blobs)

    def test_n_distinct_labels(self, X_blobs):
        r = run_kmeans(X_blobs, k=4)
        assert len(set(r["labels"])) == 4

    def test_high_silhouette_on_blobs(self, X_blobs):
        r = run_kmeans(X_blobs, k=4)
        assert r["silhouette"] > 0.80, f"Expected >0.80, got {r['silhouette']}"


class TestDBSCAN:
    def test_returns_all_keys(self, X_blobs):
        r = run_dbscan(X_blobs, eps=2.0, min_samples=5)
        for key in ["model", "labels", "silhouette", "n_clusters", "noise_count", "noise_pct"]:
            assert key in r

    def test_label_length(self, X_blobs):
        r = run_dbscan(X_blobs, eps=2.0, min_samples=5)
        assert len(r["labels"]) == len(X_blobs)

    def test_noise_pct_range(self, X_blobs):
        r = run_dbscan(X_blobs, eps=2.0, min_samples=5)
        assert 0 <= r["noise_pct"] <= 100

    def test_noise_label_is_minus_one(self, X_blobs):
        # Very tight eps forces many noise points
        r = run_dbscan(X_blobs, eps=0.01, min_samples=50)
        assert -1 in r["labels"]


class TestGMM:
    def test_returns_all_keys(self, X_blobs):
        r = run_gmm(X_blobs, n_components=4)
        for key in ["model", "labels", "proba", "silhouette", "bic", "aic"]:
            assert key in r

    def test_proba_shape(self, X_blobs):
        r = run_gmm(X_blobs, n_components=4)
        assert r["proba"].shape == (len(X_blobs), 4)

    def test_proba_sums_to_one(self, X_blobs):
        r = run_gmm(X_blobs, n_components=4)
        np.testing.assert_allclose(r["proba"].sum(axis=1), 1.0, atol=1e-6)

    def test_high_silhouette_on_blobs(self, X_blobs):
        r = run_gmm(X_blobs, n_components=4)
        assert r["silhouette"] > 0.80

    def test_bic_reasonable(self, X_blobs):
        r = run_gmm(X_blobs, n_components=4)
        assert r["bic"] < 1e8  # Sanity — not astronomically bad


class TestElbowCurve:
    def test_output_shape(self, X_blobs):
        df = elbow_curve(X_blobs, k_range=range(2, 6))
        assert len(df) == 4
        assert set(df.columns) >= {"k", "inertia", "silhouette"}

    def test_inertia_decreasing(self, X_blobs):
        df = elbow_curve(X_blobs, k_range=range(2, 7)).sort_values("k")
        inertias = df["inertia"].values
        assert all(inertias[i] >= inertias[i+1] for i in range(len(inertias)-1))


class TestLabelSegments:
    def test_segment_name_column_added(self, customers_df):
        labels = np.tile([0, 1, 2, 3], len(customers_df) // 4)
        proba  = np.random.dirichlet([1, 1, 1, 1], len(customers_df))
        result = label_segments(customers_df, labels, proba)
        assert "segment_name" in result.columns
        assert "segment_id"   in result.columns

    def test_proba_columns_added(self, customers_df):
        labels = np.tile([0, 1, 2, 3], len(customers_df) // 4)
        proba  = np.random.dirichlet([1, 1, 1, 1], len(customers_df))
        result = label_segments(customers_df, labels, proba)
        assert "gmm_p0" in result.columns
        assert "gmm_p3" in result.columns

    def test_no_rows_dropped(self, customers_df):
        labels = np.tile([0, 1, 2, 3], len(customers_df) // 4)
        proba  = np.random.dirichlet([1, 1, 1, 1], len(customers_df))
        result = label_segments(customers_df, labels, proba)
        assert len(result) == len(customers_df)
