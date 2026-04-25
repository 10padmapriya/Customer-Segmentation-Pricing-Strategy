"""
Tests for src/churn.py
"""
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.churn import compute_churn_risk, flag_high_risk, churn_summary


@pytest.fixture
def customers():
    np.random.seed(1)
    n = 200
    return pd.DataFrame({
        "customer_id":    [f"C{i}" for i in range(n)],
        "segment_name":   np.random.choice(
            ["Champions", "Loyal Customers", "At Risk", "Needs Attention"], n),
        "recency_days":   np.random.randint(1, 200, n),
        "frequency":      np.random.randint(1, 50, n),
        "monetary":       np.random.uniform(10, 5000, n),
        "avg_order_value":np.random.uniform(5, 200, n),
    })


class TestComputeChurnRisk:
    def test_output_length(self, customers):
        risk = compute_churn_risk(customers)
        assert len(risk) == len(customers)

    def test_scores_in_0_1(self, customers):
        risk = compute_churn_risk(customers)
        assert risk.between(0.01, 0.99).all()

    def test_no_nulls(self, customers):
        risk = compute_churn_risk(customers)
        assert risk.isnull().sum() == 0

    def test_high_recency_gives_high_risk(self):
        df = pd.DataFrame({
            "recency_days": [1, 200],
            "frequency":    [20, 20],
            "monetary":     [500.0, 500.0],
        })
        risk = compute_churn_risk(df)
        assert risk.iloc[1] > risk.iloc[0]

    def test_low_frequency_gives_high_risk(self):
        df = pd.DataFrame({
            "recency_days": [10, 10],
            "frequency":    [1, 40],
            "monetary":     [300.0, 300.0],
        })
        risk = compute_churn_risk(df)
        assert risk.iloc[0] > risk.iloc[1]

    def test_custom_weights_change_result(self, customers):
        default = compute_churn_risk(customers)
        custom  = compute_churn_risk(
            customers,
            weights={"recency_days": 1.0, "frequency": 0.0, "monetary": 0.0}
        )
        assert not default.equals(custom)


class TestFlagHighRisk:
    def test_threshold_applied(self, customers):
        customers["churn_risk"] = compute_churn_risk(customers)
        high = flag_high_risk(customers, threshold=0.65)
        assert (high["churn_risk"] >= 0.65).all()

    def test_sorted_descending(self, customers):
        customers["churn_risk"] = compute_churn_risk(customers)
        high = flag_high_risk(customers, threshold=0.50)
        assert high["churn_risk"].is_monotonic_decreasing


class TestChurnSummary:
    def test_returns_dataframe(self, customers):
        customers["churn_risk"] = compute_churn_risk(customers)
        summary = churn_summary(customers)
        assert isinstance(summary, pd.DataFrame)

    def test_all_segments_present(self, customers):
        customers["churn_risk"] = compute_churn_risk(customers)
        summary = churn_summary(customers)
        segs_in_data    = set(customers["segment_name"].unique())
        segs_in_summary = set(summary["segment_name"].unique())
        assert segs_in_data == segs_in_summary

    def test_avg_risk_column_exists(self, customers):
        customers["churn_risk"] = compute_churn_risk(customers)
        summary = churn_summary(customers)
        assert "avg_risk" in summary.columns
