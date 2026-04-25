"""
data/clean_data.py
==================
Cleans the raw UCI Online Retail II dataset.

Problems we fix:
  1. Missing CustomerID         — drop (can't do RFM without an ID)
  2. Cancelled invoices          — drop rows where InvoiceNo starts with 'C'
  3. Negative / zero Quantity    — drop (returns, test entries)
  4. Negative / zero UnitPrice   — drop (bad data, free samples)
  5. Bad stock codes             — drop test/postage/manual entries
  6. Duplicates                  — drop exact duplicate rows
  7. Column renaming             — standardise to snake_case
  8. Data types                  — parse dates, cast numerics

Usage:
    python data/clean_data.py
"""

from pathlib import Path
import pandas as pd


# Stock codes that are not real products (postage, bank charges, etc.)
BAD_STOCK_CODES = {
    "POST", "D", "M", "BANK CHARGES", "PADS", "DOT",
    "AMAZONFEE", "S", "CRUK", "m", "B"
}


def load_raw(path: str = "data/raw_retail.csv") -> pd.DataFrame:
    """Load the raw CSV saved by load_data.py."""
    df = pd.read_csv(
        path,
        dtype={"Customer ID": str, "StockCode": str, "Invoice": str},
        parse_dates=["InvoiceDate"],
        low_memory=False,
    )
    # Handle both column naming conventions from ucimlrepo
    df.columns = df.columns.str.strip()
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names to snake_case."""
    rename_map = {
        "Invoice":     "invoice_no",
        "InvoiceNo":   "invoice_no",
        "StockCode":   "stock_code",
        "Description": "description",
        "Quantity":    "quantity",
        "InvoiceDate": "invoice_date",
        "Price":       "unit_price",
        "UnitPrice":   "unit_price",
        "Customer ID": "customer_id",
        "CustomerID":  "customer_id",
        "Country":     "country",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def drop_missing_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no customer ID — can't segment them."""
    before = len(df)
    df = df.dropna(subset=["customer_id"])
    print(f"  Dropped missing CustomerID : {before - len(df):,} rows")
    return df


def drop_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cancelled invoices (InvoiceNo starts with 'C')."""
    before = len(df)
    df = df[~df["invoice_no"].astype(str).str.startswith("C")]
    print(f"  Dropped cancellations      : {before - len(df):,} rows")
    return df


def drop_bad_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where quantity is zero or negative."""
    before = len(df)
    df = df[df["quantity"] > 0]
    print(f"  Dropped bad quantities     : {before - len(df):,} rows")
    return df


def drop_bad_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where unit price is zero or negative."""
    before = len(df)
    df = df[df["unit_price"] > 0]
    print(f"  Dropped bad prices         : {before - len(df):,} rows")
    return df


def drop_bad_stock_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-product stock codes (postage, bank charges, etc.)."""
    before = len(df)
    df = df[~df["stock_code"].isin(BAD_STOCK_CODES)]
    # Also drop codes that are not alphanumeric (test entries)
    df = df[df["stock_code"].str.match(r"^[A-Za-z0-9]+$", na=False)]
    print(f"  Dropped bad stock codes    : {before - len(df):,} rows")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Dropped duplicates         : {before - len(df):,} rows")
    return df


def engineer_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Add a revenue column: quantity × unit_price."""
    df = df.copy()
    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct types for all columns."""
    df = df.copy()
    df["customer_id"]  = df["customer_id"].astype(str).str.strip()
    df["invoice_no"]   = df["invoice_no"].astype(str).str.strip()
    df["stock_code"]   = df["stock_code"].astype(str).str.strip()
    df["quantity"]     = df["quantity"].astype(int)
    df["unit_price"]   = df["unit_price"].astype(float)
    if not pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def clean(raw_path: str = "data/raw_retail.csv",
          out_path: str = "data/cleaned_retail.csv") -> pd.DataFrame:
    """
    Full cleaning pipeline. Runs all steps in order.

    Args:
        raw_path : Path to the raw CSV from load_data.py.
        out_path : Where to save the cleaned CSV.

    Returns:
        Clean DataFrame ready for feature engineering.
    """
    print("=" * 50)
    print("DATA CLEANING PIPELINE")
    print("=" * 50)

    df = load_raw(raw_path)
    print(f"\nRaw shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print("\nCleaning steps:")

    df = rename_columns(df)
    df = drop_missing_customers(df)
    df = drop_cancellations(df)
    df = drop_bad_quantities(df)
    df = drop_bad_prices(df)
    df = drop_bad_stock_codes(df)
    df = drop_duplicates(df)
    df = fix_dtypes(df)
    df = engineer_revenue(df)

    print(f"\nClean shape : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"Customers   : {df['customer_id'].nunique():,} unique")
    print(f"Date range  : {df['invoice_date'].min().date()} → {df['invoice_date'].max().date()}")
    print(f"Countries   : {df['country'].nunique()} unique")
    print(f"Total rev   : £{df['revenue'].sum():,.2f}")

    Path(out_path).parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return df


if __name__ == "__main__":
    clean()
