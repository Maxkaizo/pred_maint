# tasks/create_target.py

from typing import Tuple

import pandas as pd
from pandas.tseries.frequencies import to_offset
from prefect import task


def create_future_target(
    df: pd.DataFrame, gap: str = "4h", horizon: str = "2h"
) -> pd.DataFrame:
    """Creates a binary target column 'any_fail_future' based on a look-ahead window.

    This function identifies machines that will experience a failure within a
    specific future window defined by `gap` and `horizon`.

    Args:
        df: The dataset containing 'machineID', 'datetime', and 'any_fail'.
        gap: The time buffer between the current timestamp and the start of the horizon.
        horizon: The duration of the window to look for future failures.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with a new 'any_fail_future' column.
    """
    df = df.copy()
    df["any_fail_future"] = 0

    gap_offset = to_offset(gap)
    horizon_offset = to_offset(horizon)

    # For each machine, mark failures in the (t+gap, t+gap+horizon] window
    for mid, group in df.groupby("machineID"):
        fails = group.loc[group["any_fail"] == 1, "datetime"]
        if fails.empty:
            continue

        for t_fail in fails:
            window_start = t_fail - horizon_offset - gap_offset
            window_end = t_fail - gap_offset
            mask = (
                (df["machineID"] == mid)
                & (df["datetime"] >= window_start)
                & (df["datetime"] < window_end)
            )
            df.loc[mask, "any_fail_future"] = 1

    return df


@task(name="Create Future Target")
def create_targets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    gap: str = "4h",
    horizon: str = "2h",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prefect task to add the 'any_fail_future' column to train/val/test datasets.

    Args:
        df_train: The training dataset.
        df_val: The validation dataset.
        df_test: The testing dataset.
        gap: The look-ahead gap duration.
        horizon: The look-ahead horizon duration.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The enriched train, val, and test splits.
    """

    df_train = create_future_target(df_train, gap, horizon)
    df_val = create_future_target(df_val, gap, horizon)
    df_test = create_future_target(df_test, gap, horizon)

    print("✅ Target column 'any_fail_future' added to all splits")
    return df_train, df_val, df_test
