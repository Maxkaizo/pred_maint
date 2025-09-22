# tasks/feature_engineering.py

import os
import boto3
import pandas as pd
import numpy as np
from io import BytesIO
from prefect import task


# ---------------------------
# S3 Helpers
# ---------------------------
def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def load_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])


def save_parquet_to_s3(df: pd.DataFrame, bucket: str, key: str):
    s3 = s3_client()
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


# ---------------------------
# Core Feature Engineering Functions
# ---------------------------
def filter_preventive_maintenance(maintenance: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    """
    Remove reactive maintenance events by excluding records that coincide with failures.
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


def pivot_table_generic(df: pd.DataFrame, index: list, column: str, suffix: str) -> pd.DataFrame:
    """
    Generic pivot function that transforms categorical columns into binary flags.
    Always applies suffix to avoid collisions after merge.
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
    """
    Add calendar and cyclical time-based features.
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


def add_lag_rolling(df: pd.DataFrame, telemetry_cols: list) -> pd.DataFrame:
    """
    Add lag and rolling window features for telemetry signals.
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
    """
    Add global binary flags: any error, any maintenance, any failure.
    Uses suffix convention (_error, _maint, _fail).
    """
    error_cols = [c for c in df.columns if c.endswith("_error")]
    maint_cols = [c for c in df.columns if c.endswith("_maint")]
    fail_cols = [c for c in df.columns if c.endswith("_fail")]

    df["any_error"] = df[error_cols].sum(axis=1).clip(upper=1)
    df["any_maint"] = df[maint_cols].sum(axis=1).clip(upper=1)
    df["any_fail"] = df[fail_cols].sum(axis=1).clip(upper=1)
    return df


def add_recent_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling count of recent events over the past 24h.
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
    """
    Main Prefect task for feature engineering:
    - Load raw datasets from S3
    - Filter preventive maintenance
    - Pivot categorical columns into binary flags (with suffix)
    - Merge all sources into a unified dataset
    - Add time-based, telemetry, global, and recent-event features
    - Save processed dataset back to S3
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
