# tasks/compare_datasets.py (extended)
import os
import boto3
import pandas as pd
from io import BytesIO
from prefect import task

def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

def load_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))

@task(name="Compare Datasets Extended")
def compare_datasets(
    bucket: str = "datalake",
    old_key: str = "reference/processed_dataset_old.parquet",
    new_key: str = "processed/processed_dataset.parquet"
) -> None:
    """
    Compare old vs new processed datasets with extended checks:
    - Shapes and columns
    - Global flags comparison
    - Descriptive stats
    - Quality checks (nulls, types, date ranges, unique IDs)
    """
    print("📥 Loading datasets from S3...")
    old_df = load_parquet_from_s3(bucket, old_key)
    new_df = load_parquet_from_s3(bucket, new_key)

    # Shapes
    print("\n=== Shapes ===")
    print(f"Old dataset: {old_df.shape}")
    print(f"New dataset: {new_df.shape}")

    # Column differences
    old_cols = set(old_df.columns)
    new_cols = set(new_df.columns)
    print("\n=== Column Differences ===")
    print("Only in old:", old_cols - new_cols)
    print("Only in new:", new_cols - old_cols)

    # --- Global Flags ---
    global_flags = [
        "any_error", "any_maint", "any_fail",
        "any_error_last24h", "any_maint_last24h"
    ]
    print("\n=== Global Flags Comparison ===")
    for col in global_flags:
        if col in old_df.columns and col in new_df.columns:
            print(f"\nFlag: {col}")
            print("Old (mean, sum):", old_df[col].mean(), old_df[col].sum())
            print("New (mean, sum):", new_df[col].mean(), new_df[col].sum())
        else:
            print(f"⚠️ Flag {col} missing in one dataset")

    # --- Descriptive Stats ---
    print("\n=== Descriptive Statistics (first 15 numeric cols) ===")
    print("\n--- Old Dataset ---")
    print(old_df.describe().T.head(15))
    print("\n--- New Dataset ---")
    print(new_df.describe().T.head(15))

    # --- Quality Checks ---
    print("\n=== Quality Checks ===")

    def qc_report(df, label):
        print(f"\n--- {label} ---")
        print("Null counts (top 10):")
        print(df.isnull().sum().sort_values(ascending=False).head(10))
        print("\nData types:")
        print(df.dtypes.value_counts())
        if "datetime" in df.columns:
            print("Datetime range:", df["datetime"].min(), "to", df["datetime"].max())
        if "machineID" in df.columns:
            print("Unique machines:", df["machineID"].nunique())

    qc_report(old_df, "Old Dataset")
    qc_report(new_df, "New Dataset")
