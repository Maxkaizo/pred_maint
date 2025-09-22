# tasks/train_lightgbm_hyperopt.py

import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import average_precision_score, f1_score
from prefect import task

RANDOM_SEED = 42

@task(name="Train LightGBM with Hyperopt")
def train_lightgbm_hyperopt(df_train, df_val, max_evals: int = 30):
    """
    Hyperparameter tuning for LightGBM with Hyperopt.
    Logs every trial as a separate MLflow run (nested).
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

    X_train, y_train = df_train[features], df_train["any_fail_future"]
    X_val, y_val     = df_val[features], df_val["any_fail_future"]

    # Define objective function for Hyperopt
    def objective_lgb(params):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "seed": RANDOM_SEED,
            "learning_rate": params["learning_rate"],
            "num_leaves": int(params["num_leaves"]),
            "min_child_samples": int(params["min_child_samples"]),
            "feature_fraction": params["feature_fraction"],
            "feature_pre_filter": False,  # to avoid LightGBM warning
        }

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val   = lgb.Dataset(X_val, label=y_val)

        # Open a nested run for each trial
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)

            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_val],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )

            y_pred = model.predict(X_val, num_iteration=model.best_iteration)

            # Metrics
            ap = average_precision_score(y_val, y_pred)
            y_pred_binary = (y_pred > 0.5).astype(int)
            f1 = f1_score(y_val, y_pred_binary)

            mlflow.log_metric("avg_precision", ap)
            mlflow.log_metric("f1_score", f1)

            return {"loss": -ap, "status": STATUS_OK, "f1": f1}

    # Define search space
    search_space = {
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
        "num_leaves": hp.quniform("num_leaves", 16, 128, 1),
        "min_child_samples": hp.quniform("min_child_samples", 5, 50, 1),
        "feature_fraction": hp.uniform("feature_fraction", 0.6, 1.0),
    }

    trials = Trials()

    # Main parent run
    with mlflow.start_run(run_name="lightgbm_hyperopt", experiment_id=mlflow.set_experiment("pred_maintenance").experiment_id):
        best = fmin(
            fn=objective_lgb,
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(RANDOM_SEED)
        )
        mlflow.log_metric("best_ap", -min(trials.losses()))
        mlflow.log_metric("best_f1", max([t['result']['f1'] for t in trials.trials if 'result' in t]))
        mlflow.log_params(best)

    return best
