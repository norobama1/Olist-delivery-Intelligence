# src/features.py

"""
Engineer 11 delivery-delay features across four categories:
temporal, physical , logistics and calender

Usage: 
src/load_data.py
src/feature.py

Input: data/olist.merged.csv
Output: data/olist.features.csv
"""

from pandas import errors
import os
import pandas as pd

DATA_DIR = "data"

FEATURE_COLS = [
    'shipping_days',
    'estimated_days',
    'approval_hours',
    'purchase_dayofweek',
    'purchase_month',
    'product_weight_g',
    'product_volume_cm3',
    'order_value',
    'freight_value',
    'zip_distance_proxy',
    'is_peak_delayed_period',
]

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df['shipping_days'] = ((df['order_delivered_customer_date']
                            - df['order_delivered_carrier_date']).dt.total_seconds()/86400).astype(float)

    df['estimated_days'] = ((df['order_estimated_delivery_date']
                            - df['order_purchase_timestamp']).dt.total_seconds()/86400).astype(float)

    df['approval_hours'] = ((df['order_approved_at']
                            - df['order_purchase_timestamp']).dt.total_seconds() / 3600).astype(float)

    df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    return df

def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["product_weight_g", "product_height_cm", "product_length_cm", "product_width_cm"]:
        df[col] = df[col].fillna(df[col].median())
    df["product_volume_cm3"] = (df["product_height_cm"] * df["product_length_cm"] * df["product_width_cm"])
    df["order_value"] = df["price"] + df["freight_value"]
    return df

def add_logistics_features(df: pd.DataFrame) -> pd.DataFrame:
      # Brazilian CEPs: first 2 digits encode macro-region.
    # Absolute difference gives a crude but useful distance signal.
    seller_prefix = df['seller_zip_code_prefix'].astype(str).str[:2].astype(float)
    customer_prefix = df['customer_zip_code_prefix'].astype(str).str[:2].astype(float)
    df['zip_distance_proxy'] = (seller_prefix - customer_prefix).abs()
    return df

def add_calender_features(df: pd.DataFrame) -> pd.DataFrame:
    df['is_peak_delayed_period'] = df['purchase_month'].isin([3, 11, 12]).astype(int)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    #parse date columns that arrive as string
    date_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 
    'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col],errors = "coerce")
    
    df = add_temporal_features(df)
    df = add_physical_features(df)
    df = add_logistics_features(df)
    df = add_calender_features(df)
    return df

def main():
    input_path = os.path.join(DATA_DIR, "olist.merged.csv")
    print(f"Reading {input_path}....")
    df = pd.read_csv(input_path)
    print(f"{len(df):,} rows")

    print("Engineering Features:")
    df = engineer_features(df)

    neg = (df['shipping_days'] < 0).sum()
    if neg:
        print(f"  Dropping {neg} rows with negative shipping_days (carrier/delivery timestamp errors)")
        df = df[df['shipping_days'] >= 0].reset_index(drop=True)

    missing = df[FEATURE_COLS].isnull().sum()
    missing = missing[missing >0]
    if len(missing):
        print("Missing values")
        for col,n in missing.items():
            print(f"{col},{n,}")
    else:
        print(f"  No missing values in {len(FEATURE_COLS)} feature columns")
    
    keep_cols = (
        ["order_id", "delayed", "delivery_delay_days"]
        + FEATURE_COLS
        + ["review_score", "seller_id", "customer_unique_id",
           "seller_state", "customer_state", "order_purchase_timestamp"]
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
 
    output_path = os.path.join(DATA_DIR, "olist_processed.csv")
    df[keep_cols].to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}  ({len(df):,} rows × {len(keep_cols)} cols)")
 
 
if __name__ == "__main__":
    main()