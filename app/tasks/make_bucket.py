# tasks/make_bucket.py

import boto3
import os
from prefect import task


@task(name="Make S3 Bucket")
def make_bucket(bucket_name: str):
    """
    Validate or create an S3 bucket in Localstack/AWS.
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

