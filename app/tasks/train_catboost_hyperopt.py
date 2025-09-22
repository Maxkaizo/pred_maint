# tasks/train_catboost_hyperopt.py

import mlflow
import mlflow.catboost
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import average_precision_score, f1_score
from catboost import CatBoostClassifier, Pool
from prefect import task

RANDOM_SEED = 42

@task(name="Train CatBoost with Hyperopt")
def train_catboost_hyperopt(df_train, df_val, max_evals: int = 30):
    """
    Hyperparameter tuning for CatBoost with Hyperopt.
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
    def objective_cb(params):
        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": RANDOM_SEED,
            "logging_level": "Silent",
            "iterations": 500,
            "learning_rate": params["learning_rate"],
            "depth": int(params["depth"]),
            "l2_leaf_reg": params["l2_leaf_reg"],
        }

        train_pool = Pool(X_train, y_train)
        val_pool   = Pool(X_val, y_val)

        # Nested run for each trial
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)

            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

            y_pred = model.predict_proba(X_val)[:, 1]

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
        "depth": hp.quniform("depth", 4, 10, 1),
        "l2_leaf_reg": hp.uniform("l2_leaf_reg", 1, 10),
    }

    trials = Trials()

    # Main parent run
    with mlflow.start_run(run_name="catboost_hyperopt", experiment_id=mlflow.set_experiment("pred_maintenance").experiment_id):
        best = fmin(
            fn=objective_cb,
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(RANDOM_SEED)
        )

        mlflow.log_metric("best_ap", -min(trials.losses()))
        mlflow.log_metric("best_f1", max([t["result"]["f1"] for t in trials.trials if "result" in t]))
        mlflow.log_params(best)

    return best
