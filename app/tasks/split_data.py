# tasks/split_data.py

import os
import boto3
import pandas as pd
from io import BytesIO
from prefect import task


# ---------------------------
# S3 Helpers
# ---------------------------
def s3_client():
    """Return boto3 client configured for Localstack/AWS."""
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def load_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """Load a parquet file from S3 into a DataFrame."""
    s3 = s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


# ---------------------------
# Prefect Task
# ---------------------------
@task(name="Split Dataset by Date")
def split_dataset(
    bucket: str = "datalake",
    input_key: str = "processed/processed_dataset.parquet",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the processed dataset into train, validation, and test sets
    using hardcoded date ranges (time-based split).
    Returns the three splits directly as pandas DataFrames.
    """

    # Load dataset
    df = load_parquet_from_s3(bucket, input_key)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # --- Hardcoded split dates ---
    train_end = "2015-08-31"
    val_end = "2015-10-31"
    # Test: everything after val_end

    # Train: all records up to train_end
    df_train = df[df["datetime"] <= train_end].copy()

    # Validation: (train_end, val_end]
    df_val = df[(df["datetime"] > train_end) & (df["datetime"] <= val_end)].copy()

    # Test: all records after val_end
    df_test = df[df["datetime"] > val_end].copy()

    print(f"✅ Split completed | Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")

    return {"train": df_train, "val": df_val, "test": df_test}
