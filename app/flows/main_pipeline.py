# flows/main_pipeline.py

from prefect import flow
from tasks.create_future_target import create_targets
from tasks.download_data import download_data
from tasks.feature_engineering import feature_engineering
from tasks.make_bucket import make_bucket
from tasks.post_signal import post_signal
from tasks.split_data import split_dataset
from tasks.train_catboost_final import train_catboost_final
from tasks.train_catboost_hyperopt import train_catboost_hyperopt
from tasks.train_lightgbm_hyperopt import train_lightgbm_hyperopt


@flow(name="main_pipeline")
def main_pipeline() -> None:
    """Orchestrates the end-to-end Machine Learning pipeline.

    This flow covers:
    - S3 Bucket creation.
    - Data ingestion from Kaggle.
    - Feature engineering.
    - Time-based data splitting.
    - Future failure target creation.
    - Hyperparameter optimization for LightGBM and CatBoost.
    - Final model training and registration.
    - Ready signal posting for inference.
    """
    # Step 1: Ensure buckets exist
    make_bucket("datalake")
    make_bucket("artifacts")
    make_bucket("mlflow")
    make_bucket("mlflow-signals")

    # Step 2: Download dataset and upload to datalake
    dataset_uri = download_data()
    print(f"✅ Dataset available at: {dataset_uri}")

    # Step 3: Run feature engineering
    processed_uri = feature_engineering()
    print(f"✅ Processed dataset available at: {processed_uri}")

    # Step 4: Split dataset (in memory DataFrames)
    splits = split_dataset()
    df_train, df_val, df_test = splits["train"], splits["val"], splits["test"]
    print(
        f"✅ Split completed | Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}"
    )

    # Step 5: Create future target
    df_train, df_val, df_test = create_targets(df_train, df_val, df_test)
    print("✅ Targets created (any_fail_future added)")

    # Step 6: Train LightGBM with Hyperopt
    best_lgb = train_lightgbm_hyperopt(df_train, df_val)
    print(f"✅ LightGBM best params: {best_lgb}")

    # Step 7: Train CatBoost with Hyperopt
    best_cb = train_catboost_hyperopt(df_train, df_val)
    print(f"✅ CatBoost best params: {best_cb}")

    # Step 8: Train final CatBoost on full_train and evaluate on test
    final_cb = train_catboost_final(df_train, df_val, df_test, best_cb)
    print(f"✅ Final CatBoost trained & logged: {final_cb}")

    # Step 9: Post ready signal
    signal_uri = post_signal("catboost_pred_maintenance", final_cb["version"])
    print(f"✅ Ready signal posted at {signal_uri}")


if __name__ == "__main__":
    # Serve as deployment (for Prefect UI / CLI)
    main_pipeline.serve(
        name="pred-maintenance-pipeline",
        tags=["predictive_maintenance"],
        interval=None
    )
