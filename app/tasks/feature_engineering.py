# tasks/feature_engineering.py

import os
from io import BytesIO
from typing import Any, List

import boto3
import numpy as np
import pandas as pd
from prefect import task


# ---------------------------
# S3 Helpers
# ---------------------------
def s3_client() -> Any:
    """Initializes and returns a boto3 S3 client using environment variables.

    Returns:
        Any: A boto3 S3 client instance.
    """
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def load_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """Loads a CSV file from an S3 bucket into a pandas DataFrame.

    Args:
        bucket: The name of the S3 bucket.
        key: The S3 key (path) to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    s3 = s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])


def save_parquet_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    """Saves a pandas DataFrame to an S3 bucket in Parquet format.

    Args:
        df: The DataFrame to save.
        bucket: The name of the S3 bucket.
        key: The S3 key (path) where the file will be saved.
    """
    s3 = s3_client()
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


# ---------------------------
# Core Feature Engineering Functions
# ---------------------------
def filter_preventive_maintenance(maintenance: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    """Removes reactive maintenance events by excluding records that coincide with failures.

    Args:
        maintenance: The maintenance events dataset.
        failures: The failures dataset.

    Returns:
        pd.DataFrame: A filtered dataset containing only proactive maintenance.
    """
    maint_with_flag = maintenance.merge(
        failures[["datetime", "machineID", "failure"]],
        left_on=["datetime", "machineID", "comp"],
        right_on=["datetime", "machineID", "failure"],
        how="left",
        indicator=True,
    )
    return (
        maint_with_flag[maint_with_flag["_merge"] == "left_only"]
        .drop(columns=["_merge", "failure"])
        .drop_duplicates()
    )


def pivot_table_generic(df: pd.DataFrame, index: List[str], column: str, suffix: str) -> pd.DataFrame:
    """Transforms categorical columns into binary flags using a pivot table.

    Args:
        df: The source DataFrame.
        index: List of column names to use as index for the pivot.
        column: The categorical column to pivot.
        suffix: A suffix to add to the new binary columns.

    Returns:
        pd.DataFrame: The pivoted DataFrame with binary flags.
    """
    pivot = (
        df.assign(flag=1)
        .pivot_table(index=index, columns=column, values="flag", fill_value=0)
        .add_suffix(f"_{suffix}")
        .reset_index()
    )
    pivot.columns.name = None
    return pivot


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds calendar and cyclical time-based features to the DataFrame.

    Args:
        df: The dataset containing a 'datetime' column.

    Returns:
        pd.DataFrame: The dataset enriched with time-based features.
    """
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


def add_lag_rolling(df: pd.DataFrame, telemetry_cols: List[str]) -> pd.DataFrame:
    """Adds lag and rolling window features for telemetry signals.

    Args:
        df: The telemetry dataset.
        telemetry_cols: List of sensor columns to process.

    Returns:
        pd.DataFrame: The dataset enriched with lag and rolling features.
    """
    df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    for col in telemetry_cols:
        df[f"{col}_lag1"] = df.groupby("machineID")[col].shift(1)
        df[f"{col}_mean24h"] = (
            df.groupby("machineID")[col]
            .transform(lambda x: x.rolling(window=24, min_periods=1).mean())
        )
        df[f"{col}_std24h"] = (
            df.groupby("machineID")[col]
            .transform(lambda x: x.rolling(window=24, min_periods=1).std())
        )
    return df


def add_global_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Adds global binary flags indicating if any error, maintenance, or failure occurred.

    Args:
        df: The merged dataset.

    Returns:
        pd.DataFrame: The dataset enriched with global flags.
    """
    error_cols = [c for c in df.columns if c.endswith("_error")]
    maint_cols = [c for c in df.columns if c.endswith("_maint")]
    fail_cols = [c for c in df.columns if c.endswith("_fail")]

    df["any_error"] = df[error_cols].sum(axis=1).clip(upper=1)
    df["any_maint"] = df[maint_cols].sum(axis=1).clip(upper=1)
    df["any_fail"] = df[fail_cols].sum(axis=1).clip(upper=1)
    return df


def add_recent_events(df: pd.DataFrame) -> pd.DataFrame:
    """Adds rolling counts of recent events (errors, maintenance) over the past 24h.

    Args:
        df: The dataset with global flags.

    Returns:
        pd.DataFrame: The dataset enriched with recent event counts.
    """
    df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    df["any_error_last24h"] = (
        df.groupby("machineID")["any_error"]
        .transform(lambda x: x.rolling(window=24, min_periods=1).sum())
    )
    df["any_maint_last24h"] = (
        df.groupby("machineID")["any_maint"]
        .transform(lambda x: x.rolling(window=24, min_periods=1).sum())
    )
    return df


# ---------------------------
# Prefect Task
# ---------------------------
@task(name="Feature Engineering")
def feature_engineering(
    bucket: str = "datalake",
    input_prefix: str = "raw",
    output_prefix: str = "processed",
) -> str:
    """Main Prefect task for the feature engineering pipeline.

    This task orchestrates:
    - Data loading from S3.
    - Filtering of reactive maintenance.
    - Pivoting of categorical events into binary flags.
    - Integration of telemetry with event flags.
    - Creation of time-based and rolling window features.
    - Persistence of the final processed dataset to S3 in Parquet format.

    Args:
        bucket: The name of the S3 bucket where raw data is stored.
        input_prefix: The S3 prefix for raw data files.
        output_prefix: The S3 prefix where the processed dataset will be saved.

    Returns:
        str: The S3 URI of the processed dataset.
    """
    # Load raw datasets
    telemetry = load_csv_from_s3(bucket, f"{input_prefix}/PdM_telemetry.csv")
    errors = load_csv_from_s3(bucket, f"{input_prefix}/PdM_errors.csv")
    maintenance = load_csv_from_s3(bucket, f"{input_prefix}/PdM_maint.csv")
    failures = load_csv_from_s3(bucket, f"{input_prefix}/PdM_failures.csv")
    machines = load_csv_from_s3(bucket, f"{input_prefix}/PdM_machines.csv")

    # Parse datetime
    for df in [telemetry, errors, maintenance, failures]:
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Filter proactive maintenance
    maintenance = filter_preventive_maintenance(maintenance, failures)

    # Pivots with suffixes
    errors_pvt = pivot_table_generic(errors, ["datetime", "machineID"], "errorID", "error")
    maint_pvt = pivot_table_generic(maintenance, ["datetime", "machineID"], "comp", "maint")
    fails_pvt = pivot_table_generic(failures, ["datetime", "machineID"], "failure", "fail")
    machines_pvt = pivot_table_generic(machines, ["machineID"], "model", "model")

    # Merge everything
    full_df = telemetry.merge(errors_pvt, on=["datetime", "machineID"], how="left") \
                       .merge(maint_pvt, on=["datetime", "machineID"], how="left") \
                       .merge(fails_pvt, on=["datetime", "machineID"], how="left") \
                       .merge(machines_pvt, on=["machineID"], how="left")

    # Add features
    full_df = add_time_features(full_df)
    full_df = add_lag_rolling(full_df, ["volt", "rotate", "pressure", "vibration"])
    full_df = add_global_flags(full_df)
    full_df = add_recent_events(full_df)

    # Fill NaN from pivots with 0
    full_df = full_df.fillna(0)

    # Save processed dataset
    output_key = f"{output_prefix}/processed_dataset.parquet"
    save_parquet_to_s3(full_df, bucket, output_key)

    return f"s3://{bucket}/{output_key}"
