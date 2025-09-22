# tasks/download_data.py

import os
import shutil
import kagglehub
import boto3
from prefect import task


@task(name="Download and Upload Kaggle Dataset")
def download_data(
    dataset: str = "arnabbiswas1/microsoft-azure-predictive-maintenance",
    bucket: str = "datalake",
    prefix: str = "raw"
) -> str:
    """
    Download a Kaggle dataset and upload it to an S3 bucket (Localstack/AWS).

    Args:
        dataset (str): Kaggle dataset identifier.
        bucket (str): S3 bucket name to upload dataset.
        prefix (str): Prefix inside the bucket (acts like folder).

    Returns:
        str: S3 URI where the dataset is stored.
    """
    print(f"📥 Downloading dataset: {dataset}")
    path = kagglehub.dataset_download(dataset)

    # Init S3 client (use env vars for credentials + endpoint)
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),       # Use localstack instead of real AWS service
    )

    # Walk through KaggleHub cache and upload each file
    for root, _, files in os.walk(path):
        for file in files:
            local_path = os.path.join(root, file)
            s3_key = f"{prefix}/{file}"

            s3.upload_file(local_path, bucket, s3_key)
            print(f"⬆️ Uploaded {file} to s3://{bucket}/{s3_key}")

    return f"s3://{bucket}/{prefix}/"
