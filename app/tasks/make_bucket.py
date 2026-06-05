# tasks/make_bucket.py

import os

import boto3
from prefect import task


@task(name="Make S3 Bucket")
def make_bucket(bucket_name: str) -> str:
    """Validates or creates an S3 bucket.

    Args:
        bucket_name: The name of the bucket to ensure exists.

    Returns:
        str: A message indicating the status of the bucket.
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),       # Use localstack instead of real AWS service
    )

    existing_buckets = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]

    if bucket_name in existing_buckets:
        return f"✅ Bucket '{bucket_name}' already exists."

    s3.create_bucket(Bucket=bucket_name)
    return f"🆕 Bucket '{bucket_name}' created successfully."

