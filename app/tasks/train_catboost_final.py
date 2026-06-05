# tasks/train_catboost_final.py

from typing import Any, Dict

import mlflow
import mlflow.catboost
import pandas as pd
from catboost import CatBoostClassifier
from mlflow.tracking import MlflowClient
from prefect import task
from sklearn.metrics import average_precision_score, classification_report, f1_score

RANDOM_SEED = 42


@task(name="Train Final CatBoost Model")
def train_catboost_final(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, best_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Trains the final CatBoost model using train + validation sets and evaluates on test.

    This task:
    - Combines train and validation sets for final training.
    - Trains a CatBoostClassifier with optimized hyperparameters.
    - Evaluates performance on the held-out test set.
    - Logs parameters, metrics (AP, F1), and artifacts to MLflow.
    - Registers the model in the MLflow Model Registry and tags it as 'Staging'.

    Args:
        df_train: The training dataset.
        df_val: The validation dataset.
        df_test: The testing dataset.
        best_params: Dictionary of optimized hyperparameters.

    Returns:
        Dict[str, Any]: A dictionary containing metrics and MLflow run/version info.
    """

    # ---------------------------
    # Feature Selection
    # ---------------------------
    exclude_cols = [
        "datetime",
        "machineID",
        "any_fail",
        "any_fail_future",
    ] + [c for c in df_train.columns if c.endswith("_fail")]

    features = [c for c in df_train.columns if c not in exclude_cols]

    # Build full train and test splits
    full_train = pd.concat([df_train, df_val], axis=0)
    X_full, y_full = full_train[features], full_train["any_fail_future"]
    X_test, y_test = df_test[features], df_test["any_fail_future"]

    # Convert hyperopt params into CatBoost-accepted params
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",   # CatBoost metric, but we'll log AP & F1 manually
        "verbose": 100,
        "random_seed": RANDOM_SEED,
        "iterations": 1000,
        "early_stopping_rounds": 50,
        # Cast hyperopt params to correct types
        "depth": int(best_params.get("depth", 6)),
        "learning_rate": float(best_params.get("learning_rate", 0.1)),
        "l2_leaf_reg": float(best_params.get("l2_leaf_reg", 3.0)),
    }

    # Start MLflow run (final model)
    with mlflow.start_run(run_name="catboost_final", experiment_id=mlflow.set_experiment("pred_maintenance").experiment_id) as run:
        mlflow.log_params(params)

        # Train
        model = CatBoostClassifier(**params)
        model.fit(X_full, y_full)

        # Evaluate on test
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        ap = average_precision_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("test_avg_precision", ap)
        mlflow.log_metric("test_f1_score", f1)

        # Save classification report as text artifact
        report = classification_report(y_test, y_pred, digits=4)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # Log final model and promote it
        #mlflow.catboost.log_model(
        #    model,
        #    artifact_path="model",
        #    registered_model_name="catboost_pred_maintenance"
        #)


        # Log model as artifact
        mlflow.catboost.log_model(model, artifact_path="model")

        # Register model explicitly
        result = mlflow.register_model(
            f"runs:/{run.info.run_id}/model",
            "catboost_pred_maintenance"
        )

        # Add a stage tag
        client = MlflowClient()
        client.set_model_version_tag(
            name="catboost_pred_maintenance",
            version=result.version,
            key="stage",
            value="Staging"
        )

        print("✅ Final CatBoost model trained and promoted")
        print(f"📊 Test AP: {ap:.4f}, Test F1: {f1:.4f}")
        print(f"🔗 View run at: {mlflow.get_tracking_uri()}")

    return {"ap": ap, "f1": f1, "run_id": run.info.run_id, "version": result.version}
