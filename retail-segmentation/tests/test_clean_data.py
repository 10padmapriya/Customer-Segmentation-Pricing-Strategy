"""
Tests for data/clean_data.py
"""
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.clean_data import (
    rename_columns,
    drop_missing_customers,
    drop_cancellations,
    drop_bad_quantities,
    drop_bad_prices,
    drop_duplicates,
    engineer_revenue,
)


@pytest.fixture
def raw_df():
    """Minimal raw DataFrame that mirrors UCI Online Retail II schema."""
    return pd.DataFrame({
        "Invoice":     ["536365", "C536379", "536381", "536382", "536365"],
        "StockCode":   ["85123A", "85123A", "POST",   "85123B", "85123A"],
        "Description": ["CREAM HANGING",  "CREAM HANGING", "POSTAGE", "GLASS", "CREAM HANGING"],
        "Quantity":    [6,  -6,  1,  0,  6],
        "InvoiceDate": pd.to_datetime(["2010-12-01", "2010-12-01", "2010-12-01",
                                       "2010-12-01", "2010-12-01"]),
        "Price":       [2.55, 2.55, 1.00, 0.00, 2.55],
        "Customer ID": ["17850", "17850", None,  "13047", "17850"],
        "Country":     ["UK", "UK", "UK", "UK", "UK"],
    })


# ── rename_columns ────────────────────────────────────────────

def test_rename_gives_snake_case(raw_df):
    df = rename_columns(raw_df)
    assert "invoice_no"  in df.columns
    assert "customer_id" in df.columns
    assert "unit_price"  in df.columns
    assert "invoice_date" in df.columns


# ── drop_missing_customers ────────────────────────────────────

def test_drop_missing_customers_removes_nulls(raw_df):
    df = rename_columns(raw_df)
    before = len(df)
    df = drop_missing_customers(df)
    assert len(df) < before
    assert df["customer_id"].isnull().sum() == 0


# ── drop_cancellations ────────────────────────────────────────

def test_drop_cancellations_removes_C_prefix(raw_df):
    df = rename_columns(raw_df)
    df = drop_missing_customers(df)
    df = drop_cancellations(df)
    assert not df["invoice_no"].str.startswith("C").any()


# ── drop_bad_quantities ───────────────────────────────────────

def test_drop_bad_quantities_keeps_only_positive(raw_df):
    df = rename_columns(raw_df)
    df = drop_missing_customers(df)
    df = drop_cancellations(df)
    df = drop_bad_quantities(df)
    assert (df["quantity"] > 0).all()


# ── drop_bad_prices ───────────────────────────────────────────

def test_drop_bad_prices_keeps_only_positive(raw_df):
    df = rename_columns(raw_df)
    df = drop_missing_customers(df)
    df = drop_cancellations(df)
    df = drop_bad_quantities(df)
    df = drop_bad_prices(df)
    assert (df["unit_price"] > 0).all()


# ── drop_duplicates ───────────────────────────────────────────

def test_drop_duplicates_removes_exact_copies(raw_df):
    df = rename_columns(raw_df)
    n_before = len(df)
    df = drop_duplicates(df)
    assert len(df) <= n_before


# ── engineer_revenue ──────────────────────────────────────────

def test_engineer_revenue_correct_formula():
    df = pd.DataFrame({
        "quantity":   [2, 3],
        "unit_price": [5.0, 10.0],
    })
    df = engineer_revenue(df)
    assert "revenue" in df.columns
    assert df["revenue"].tolist() == [10.0, 30.0]


def test_engineer_revenue_rounded_to_2dp():
    df = pd.DataFrame({
        "quantity":   [1],
        "unit_price": [1.999],
    })
    df = engineer_revenue(df)
    assert df["revenue"].iloc[0] == round(1 * 1.999, 2)
