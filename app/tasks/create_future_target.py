# tasks/create_target.py

import pandas as pd
from pandas.tseries.frequencies import to_offset
from prefect import task


def create_future_target(df: pd.DataFrame, gap="4h", horizon="2h") -> pd.DataFrame:
    """
    Creates a binary target column 'any_fail_future' that indicates
    if a machine will fail in the near future.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['machineID', 'datetime', 'any_fail'].
    gap : str
        Time to skip after current timestamp before starting horizon (e.g., '4h').
    horizon : str
        Time window after gap in which to look for failures (e.g., '2h').

    Returns
    -------
    pd.DataFrame
        Copy of df with new column 'any_fail_future'.
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
            window_end   = t_fail - gap_offset
            mask = (df["machineID"] == mid) & (df["datetime"] > window_start) & (df["datetime"] <= window_end)
            df.loc[mask, "any_fail_future"] = 1

    return df


@task(name="Create Future Target")
def create_targets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    gap: str = "4h",
    horizon: str = "2h",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prefect task to add the 'any_fail_future' column to train/val/test datasets.
    """

    df_train = create_future_target(df_train, gap, horizon)
    df_val   = create_future_target(df_val, gap, horizon)
    df_test  = create_future_target(df_test, gap, horizon)

    print("✅ Target column 'any_fail_future' added to all splits")
    return df_train, df_val, df_test
