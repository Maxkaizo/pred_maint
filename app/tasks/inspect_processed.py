# tasks/inspect_processed.py

import os
import boto3
import pandas as pd
from io import BytesIO
from prefect import task


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localstack:4566"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


@task(name="Inspect Processed Dataset")
def inspect_processed(
    bucket: str = "datalake",
    key: str = "processed/processed_dataset.parquet",
    n: int = 5,
):
    """
    Load the processed dataset from S3 and print the first n rows
    transposed for easier comparison with notebook results.
    """
    s3 = s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(BytesIO(obj["Body"].read()))

    print("\n=== Transposed preview of first rows ===")
    print(df.head(n).T)
    return df.head(n).T
