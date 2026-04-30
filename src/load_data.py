# src/load_data.py
"""
Pipeline:
1. Load all CSVs into a dictionary of DataFrames
2. Merge orders → order_items → products → sellers → customers → payments
3. Filter to delivered orders (order_status == "delivered")
4. Create target: delayed = 1 if actual > estimated delivery, else 0
5. Return a clean, model-ready DataFrame

Usage:
    from src.load_data import load_all_tables
    tables = load_all_tables("data/")
    df = build_feature_table(tables)

Output:
    data/olist.merged.csv
"""

import os
from pathlib import Path
import pandas as pd

DATA_DIR = "data/"

# Canonical Olist file stems → short alias
TABLE_REGISTRY = {
    "olist_orders_dataset":                  "orders",
    "olist_order_items_dataset":             "items",
    "olist_order_payments_dataset":          "payments",
    "olist_order_reviews_dataset":           "reviews",
    "olist_products_dataset":                "products",
    "olist_sellers_dataset":                 "sellers",
    "olist_customers_dataset":               "customers",
    "product_category_name_translation":     "category_translation",
    # geolocation excluded- multiple rows per zip prefix causes row explosion
    #zip-prefix distance proxy used instead (features.py)
}

# Columns that should be parsed as datetime
DATETIME_COLS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
}

def load_raw_tables() -> dict[str, pd.DataFrame]:
    """Scan data_dir for Olist CSVs, load each into a DataFrame keyed by short alias."""
    tables = {}
    for filename, name in TABLE_REGISTRY.items():
        parse_dates = DATETIME_COLS.get(name, [])
        tables[name] = pd.read_csv(
            f"{DATA_DIR}{filename}.csv",
            parse_dates=parse_dates if parse_dates else False,
        )
    return tables

def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate payments to order level — one row per order_id."""
    payment_agg = payments.groupby("order_id").agg(
        total_payment = ("payment_value","sum"),
        payment_installments = ("payment_installments","max"),
        n_payment_method = ("payment_type","nunique"),
    ).reset_index()

    return payment_agg

def merge_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge Olist tables into a single order-item-level dataframe.

    Join path: orders → items (inner) → products, sellers, customers,
    payments (aggregated), reviews — all left joins.
    """
    df = tables["orders"].merge(tables["items"], on='order_id', how='inner')
    df = df.merge(tables["products"], on='product_id', how='left')
    df = df.merge(tables['sellers'], on='seller_id', how='left')
    df = df.merge(tables["customers"], on='customer_id', how='left')
    df = df.merge(aggregate_payments(tables["payments"]), on='order_id', how='left')
    # review_score excluded to avoid target leakage
    df = df.merge(tables["reviews"][["order_id", "review_id"]], on="order_id", how="left")
    return df

def filter_delivered(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only delivered orders with valid delivery dates"""
    df = df[df["order_status"] == "delivered"].copy()
    df = df.dropna(subset=[
        "order_delivered_customer_date","order_estimated_delivery_date"
    ])
    return df

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """ Delayed =1 if actual delivery > estimated delivery else 0"""
    df["delivery_delay_days"] = ( 
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days

    df["delayed"] = (df["delivery_delay_days"] > 0).astype(int)
    return df

def main():
    print("Loading raw tables...")
    tables = load_raw_tables()
    for name, t in tables.items():
        print(f"{name:<12s} {t.shape[0]:>7} rows")
    
    print("\nMerging....")
    df = merge_tables(tables)
    print(f" Merged: {len(df):,} rows")

    print("\nFiltering to delivered orders....")
    df = filter_delivered(df)
    print(f" Delivered: {len(df):,} rows")

    print("\nCreating target variable...")
    df = create_target(df)
    delay_rate = df["delayed"].mean() *100
    print(f" Delay rate: {delay_rate:.1f}% "
           f" ({df['delayed'].sum():,} delayed/{len(df):,} total)")
    

    output_path = os.path.join(DATA_DIR,"olist.merged.csv")
    df.to_csv(output_path,index=False)
    print(f"\nSaved merged data to {output_path}")

if __name__ == "__main__":
    main()
