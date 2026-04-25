"""
Tests for src/wtp_model.py
"""
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.wtp_model import (
    build_wtp_target, train_wtp_model,
    predict_wtp, compute_shap, shap_importance,
)


@pytest.fixture
def customers():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "customer_id":         [f"C{i}" for i in range(n)],
        "recency_days":        np.random.randint(1, 180, n),
        "frequency":           np.random.randint(1, 50, n),
        "monetary":            np.random.uniform(10, 5000, n),
        "avg_order_value":     np.random.uniform(5, 400, n),
        "avg_items_per_order": np.random.uniform(1, 20, n),
        "unique_products":     np.random.randint(1, 80, n),
        "revenue_per_day":     np.random.uniform(0.1, 50, n),
        "active_days":         np.random.randint(1, 700, n),
    })
    df["wtp_proxy"] = build_wtp_target(df)
    return df


class TestBuildWTPTarget:
    def test_output_length(self, customers):
        wtp = build_wtp_target(customers)
        assert len(wtp) == len(customers)

    def test_all_non_negative(self, customers):
        wtp = build_wtp_target(customers)
        assert (wtp >= 0).all()

    def test_no_nulls(self, customers):
        wtp = build_wtp_target(customers)
        assert wtp.isnull().sum() == 0

    def test_high_aov_gives_high_wtp(self):
        df = pd.DataFrame({
            "recency_days":    [10, 10],
            "frequency":       [10, 10],
            "monetary":        [500.0, 500.0],
            "avg_order_value": [10.0, 200.0],
        })
        wtp = build_wtp_target(df)
        assert wtp.iloc[1] > wtp.iloc[0]

    def test_high_recency_gives_lower_wtp(self):
        df = pd.DataFrame({
            "recency_days":    [1, 300],
            "frequency":       [10, 10],
            "monetary":        [500.0, 500.0],
            "avg_order_value": [50.0, 50.0],
        })
        wtp = build_wtp_target(df)
        assert wtp.iloc[0] > wtp.iloc[1]


class TestTrainWTPModel:
    def test_returns_model_features_metrics(self, customers):
        model, features, metrics = train_wtp_model(customers)
        assert model is not None
        assert isinstance(features, list)
        assert isinstance(metrics, dict)

    def test_metrics_keys(self, customers):
        _, _, metrics = train_wtp_model(customers)
        for k in ["r2_test", "mae_test", "cv_r2_mean", "cv_r2_std"]:
            assert k in metrics

    def test_r2_positive(self, customers):
        _, _, metrics = train_wtp_model(customers)
        assert metrics["r2_test"] > 0, f"R² too low: {metrics['r2_test']}"

    def test_mae_non_negative(self, customers):
        _, _, metrics = train_wtp_model(customers)
        assert metrics["mae_test"] >= 0


class TestPredictWTP:
    def test_length_matches_input(self, customers):
        model, features, _ = train_wtp_model(customers)
        preds = predict_wtp(model, customers, features)
        assert len(preds) == len(customers)

    def test_non_negative(self, customers):
        model, features, _ = train_wtp_model(customers)
        preds = predict_wtp(model, customers, features)
        assert (preds >= 0).all()

    def test_index_aligned(self, customers):
        model, features, _ = train_wtp_model(customers)
        preds = predict_wtp(model, customers, features)
        assert list(preds.index) == list(customers.index)


class TestSHAP:
    def test_shap_shape(self, customers):
        model, features, _ = train_wtp_model(customers)
        shap_vals, X_sample = compute_shap(model, customers, features, sample_n=100)
        assert shap_vals.shape == X_sample.shape

    def test_importance_sorted_descending(self, customers):
        model, features, _ = train_wtp_model(customers)
        shap_vals, X_sample = compute_shap(model, customers, features, sample_n=100)
        imp = shap_importance(shap_vals, X_sample)
        vals = imp["mean_abs_shap"].values
        assert all(vals[i] >= vals[i+1] for i in range(len(vals)-1))

    def test_all_features_in_importance(self, customers):
        model, features, _ = train_wtp_model(customers)
        shap_vals, X_sample = compute_shap(model, customers, features, sample_n=100)
        imp = shap_importance(shap_vals, X_sample)
        assert set(imp["feature"]) == set(features)
