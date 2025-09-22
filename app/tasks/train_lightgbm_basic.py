# tasks/train_lightgbm_basic.py

import mlflow
import lightgbm as lgb
import numpy as np
from sklearn.metrics import classification_report, f1_score, average_precision_score
from prefect import task


RANDOM_SEED = 42


@task(name="Train LightGBM (Basic)")
def train_lightgbm_basic(df_train, df_val, experiment_name="lightgbm_baseline"):
    """
    Train a basic LightGBM model with early stopping and log results to MLflow.

    Parameters
    ----------
    df_train, df_val : pd.DataFrame
        Must contain features + 'any_fail_future' as target.
    experiment_name : str
        MLflow experiment name.
    """

    # ----------------------------
    # Feature selection
    # ----------------------------
    exclude_cols = [
        "datetime",
        "machineID",
        "any_fail",
        "any_fail_future",
    ] + [c for c in df_train.columns if c.endswith("_fail")]

    features = [c for c in df_train.columns if c not in exclude_cols]

    X_train, y_train = df_train[features], df_train["any_fail_future"]
    X_val, y_val     = df_val[features], df_val["any_fail_future"]

    # ----------------------------
    # LightGBM Dataset
    # ----------------------------
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val, label=y_val)

    # ----------------------------
    # Parameters
    # ----------------------------
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "is_unbalance": True,
        "seed": RANDOM_SEED,
    }

    # ----------------------------
    # MLflow tracking
    # ----------------------------
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)

        # Train model
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(100),
            ],
        )

        # Predictions
        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Metrics
        f1 = f1_score(y_val, y_pred, average="binary")
        ap = average_precision_score(y_val, y_pred_proba)

        mlflow.log_metric("f1_val", f1)
        mlflow.log_metric("ap_val", ap)

        # Save model
        mlflow.lightgbm.log_model(model, "model")

        # Console output
        print("Validation performance (LightGBM):")
        print(classification_report(y_val, y_pred, digits=4))

    return model
