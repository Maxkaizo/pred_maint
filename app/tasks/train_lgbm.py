# tasks/train_lgbm.py

import os
import mlflow
import lightgbm as lgb
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import average_precision_score, f1_score
from prefect import task

RANDOM_SEED = 42


@task(name="Train and Optimize LightGBM")
def train_lgbm(X_train, y_train, X_val, y_val, max_evals: int = 30):
    """
    Train and optimize a LightGBM model using Hyperopt, logging results in MLflow.
    Selects best model based on Average Precision (AP).
    """

    # Define objective function
    def objective(params):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "seed": RANDOM_SEED,
            "learning_rate": params["learning_rate"],
            "num_leaves": int(params["num_leaves"]),
            "min_child_samples": int(params["min_child_samples"]),
            "feature_fraction": params["feature_fraction"],
        }

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val)

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)

            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_val],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            y_pred = model.predict(X_val, num_iteration=model.best_iteration)

            # Metrics
            ap = average_precision_score(y_val, y_pred)
            y_pred_labels = (y_pred >= 0.5).astype(int)
            f1 = f1_score(y_val, y_pred_labels)

            mlflow.log_metric("avg_precision", ap)
            mlflow.log_metric("f1_score", f1)

            return {"loss": -ap, "status": STATUS_OK, "model": model}

    # Define search space
    search_space = {
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
        "num_leaves": hp.quniform("num_leaves", 16, 128, 1),
        "min_child_samples": hp.quniform("min_child_samples", 5, 50, 1),
        "feature_fraction": hp.uniform("feature_fraction", 0.6, 1.0),
    }

    trials = Trials()
    with mlflow.start_run(run_name="LightGBM_Optimization"):
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(RANDOM_SEED),
        )

        # Get best trial
        best_trial = min(trials.results, key=lambda x: x["loss"])
        best_model = best_trial["model"]

        # Log best model
        mlflow.lightgbm.log_model(best_model, artifact_path="model")

        mlflow.log_params(best)
        print("Best LightGBM params:", best)

    return best
