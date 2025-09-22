# wait_for_model.py
import boto3
import os
import time
import sys

MODEL_NAME = "catboost_pred_maintenance"
SIGNAL_BUCKET = "mlflow-signals"

s3 = boto3.client("s3", endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"))

while True:
    try:
        objects = s3.list_objects_v2(Bucket=SIGNAL_BUCKET, Prefix=f"{MODEL_NAME}/ready-")
        if "Contents" in objects and len(objects["Contents"]) > 0:
            print(f"✅ Found ready signal for {MODEL_NAME}")
            sys.exit(0)   # ✅ explicitly exit success
    except Exception as e:
        print(f"⚠️ Waiting for signal bucket: {e}")

    print("⏳ No signal yet, retrying in 15s...")
    time.sleep(15)
