# tasks/post_signal.py

import boto3
import os
from prefect import task

@task(name="Post Ready Signal")
def post_signal(model_name: str, version: str):
    """
    Uploads a signal file to S3/Localstack after training completes.
    Inference service will watch for this file.
    """
    s3 = boto3.client("s3", endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"))
    bucket = "mlflow-signals"
    key = f"{model_name}/ready-v{version}.txt"

    # Ensure bucket exists
    try:
        s3.create_bucket(Bucket=bucket)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass
    except s3.exceptions.BucketAlreadyExists:
        pass

    body = f"Model {model_name} v{version} is ready".encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body)

    print(f"✅ Signal posted: s3://{bucket}/{key}")
    return f"s3://{bucket}/{key}"
