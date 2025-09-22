import os
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow.tracking import MlflowClient

# MLflow model registry info
MODEL_NAME = "catboost_pred_maintenance"
MODEL_STAGE_TAG = "Staging"

# ---------------------------
# Load model by tag
# ---------------------------
def load_model_by_tag(model_name: str, tag_value: str):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    staging_versions = [v for v in versions if v.tags.get("stage") == tag_value]

    if not staging_versions:
        raise RuntimeError(f"No versions of {model_name} found with tag stage={tag_value}")

    latest = max(staging_versions, key=lambda v: int(v.version))
    model_uri = f"models:/{model_name}/{latest.version}"
    print(f"📂 Loading model from {model_uri} (tag={tag_value})")
    return mlflow.pyfunc.load_model(model_uri)

# Load the model once at startup
model = load_model_by_tag(MODEL_NAME, MODEL_STAGE_TAG)

app = FastAPI(title="Predictive Maintenance Inference API")

# --- Define schema with engineered features ---
class Features(BaseModel):
    volt: float
    rotate: float
    pressure: float
    vibration: float
    error1_error: int
    error2_error: int
    error3_error: int
    error4_error: int
    error5_error: int
    comp1_maint: int
    comp2_maint: int
    comp3_maint: int
    comp4_maint: int
    model1_model: int
    model2_model: int
    model3_model: int
    model4_model: int
    hour: int
    dayofweek: int
    month: int
    hour_sin: float
    hour_cos: float
    dayofweek_sin: float
    dayofweek_cos: float
    volt_lag1: float
    volt_mean24h: float
    volt_std24h: float
    rotate_lag1: float
    rotate_mean24h: float
    rotate_std24h: float
    pressure_lag1: float
    pressure_mean24h: float
    pressure_std24h: float
    vibration_lag1: float
    vibration_mean24h: float
    vibration_std24h: float
    any_error: int
    any_maint: int
    any_error_last24h: int
    any_maint_last24h: int

@app.post("/predict")
def predict(data: Features):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Predict probability of future failure
    proba = model.predict(df)[0]

    return {
        "failure_probability": float(proba),
        "decision": "dispatch_tech" if proba > 0.5 else "no_action"
    }
