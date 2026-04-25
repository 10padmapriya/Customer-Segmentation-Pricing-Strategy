"""
Tests for src/features.py
"""
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import build_customer_features, scale_features, CLUSTER_FEATURES


@pytest.fixture
def clean_df():
    """Minimal cleaned transaction DataFrame."""
    today = datetime(2011, 12, 10)
    return pd.DataFrame({
        "customer_id": ["A", "A", "A", "B", "B", "C"],
        "invoice_no":  ["1", "1", "2", "3", "4", "5"],
        "invoice_date": [
            today - timedelta(days=5),
            today - timedelta(days=5),
            today - timedelta(days=3),
            today - timedelta(days=60),
            today - timedelta(days=58),
            today - timedelta(days=1),
        ],
        "quantity":    [2,  3,  1,  5,  2,  10],
        "unit_price":  [5.0, 10.0, 3.0, 2.0, 8.0, 1.5],
        "revenue":     [10.0, 30.0, 3.0, 10.0, 16.0, 15.0],
        "stock_code":  ["AAA", "BBB", "AAA", "CCC", "DDD", "EEE"],
        "country":     ["UK", "UK", "UK", "UK", "UK", "FR"],
    })


class TestBuildCustomerFeatures:
    def test_one_row_per_customer(self, clean_df):
        customers = build_customer_features(clean_df)
        assert len(customers) == clean_df["customer_id"].nunique()

    def test_required_columns_exist(self, clean_df):
        customers = build_customer_features(clean_df)
        required = ["customer_id", "recency_days", "frequency",
                    "monetary", "r_score", "f_score", "m_score", "rfm_total"]
        for col in required:
            assert col in customers.columns, f"Missing: {col}"

    def test_recency_is_non_negative(self, clean_df):
        customers = build_customer_features(clean_df)
        assert (customers["recency_days"] >= 0).all()

    def test_frequency_counts_unique_invoices(self, clean_df):
        customers = build_customer_features(clean_df)
        # Customer A has 2 unique invoices
        cust_a = customers.loc[customers["customer_id"] == "A", "frequency"].iloc[0]
        assert cust_a == 2

    def test_monetary_is_sum_of_revenue(self, clean_df):
        customers = build_customer_features(clean_df)
        # Customer A: 10 + 30 + 3 = 43
        cust_a = customers.loc[customers["customer_id"] == "A", "monetary"].iloc[0]
        assert abs(cust_a - 43.0) < 0.01

    def test_rfm_scores_in_range_1_to_5(self, clean_df):
        customers = build_customer_features(clean_df)
        for col in ["r_score", "f_score", "m_score"]:
            assert customers[col].between(1, 5).all(), f"{col} out of [1,5]"

    def test_rfm_total_in_range(self, clean_df):
        customers = build_customer_features(clean_df)
        assert customers["rfm_total"].between(3, 15).all()

    def test_avg_order_value_positive(self, clean_df):
        customers = build_customer_features(clean_df)
        assert (customers["avg_order_value"] > 0).all()

    def test_high_recency_gets_low_r_score(self, clean_df):
        """Customer B hasn't bought in 60 days → lower r_score than C (1 day)."""
        customers = build_customer_features(clean_df)
        r_b = customers.loc[customers["customer_id"] == "B", "r_score"].iloc[0]
        r_c = customers.loc[customers["customer_id"] == "C", "r_score"].iloc[0]
        assert r_b <= r_c, "Dormant customer should have ≤ r_score than recent one"


class TestScaleFeatures:
    def test_output_shape(self, clean_df):
        customers = build_customer_features(clean_df)
        cols = ["recency_days", "frequency", "monetary"]
        X, scaler = scale_features(customers, cols)
        assert X.shape == (len(customers), len(cols))

    def test_zero_mean(self, clean_df):
        customers = build_customer_features(clean_df)
        cols = ["recency_days", "frequency", "monetary"]
        X, _ = scale_features(customers, cols)
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-10)

    def test_scaler_returned(self, clean_df):
        customers = build_customer_features(clean_df)
        cols = ["recency_days", "frequency"]
        _, scaler = scale_features(customers, cols)
        assert hasattr(scaler, "inverse_transform")
