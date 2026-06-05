# tasks/inspect_processed.py

from io import BytesIO
from typing import Any

import pandas as pd
from prefect import task


def s3_client() -> Any:
    """Initializes and returns a boto3 S3 client.

    Returns:
        Any: A boto3 S3 client instance.
    """
...
@task(name="Inspect Processed Dataset")
def inspect_processed(
    bucket: str = "datalake",
    key: str = "processed/processed_dataset.parquet",
    n: int = 5,
) -> pd.DataFrame:
    """Loads and prints a transposed preview of the processed dataset.

    Args:
        bucket: The name of the S3 bucket.
        key: The S3 key (path) to the processed dataset.
        n: Number of rows to preview.

    Returns:
        pd.DataFrame: A transposed preview of the first n rows.
    """
    s3 = s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(BytesIO(obj["Body"].read()))

    print("\n=== Transposed preview of first rows ===")
    print(df.head(n).T)
    return df.head(n).T
