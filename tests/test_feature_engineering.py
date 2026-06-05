import pandas as pd

from app.tasks.feature_engineering import (
    add_global_flags,
    add_time_features,
    filter_preventive_maintenance,
    pivot_table_generic,
)


def test_filter_preventive_maintenance() -> None:
    maint_df = pd.DataFrame({
        "datetime": ["2015-01-01", "2015-01-02"],
        "machineID": [1, 1],
        "comp": ["comp1", "comp2"]
    })
    maint_df["datetime"] = pd.to_datetime(maint_df["datetime"])

    failures_df = pd.DataFrame({
        "datetime": ["2015-01-02"],
        "machineID": [1],
        "failure": ["comp2"]
    })
    failures_df["datetime"] = pd.to_datetime(failures_df["datetime"])

    result = filter_preventive_maintenance(maint_df, failures_df)

    # Should only keep comp1 from 2015-01-01
    assert len(result) == 1
    assert result.iloc[0]["comp"] == "comp1"

def test_pivot_table_generic() -> None:
    df = pd.DataFrame({
        "machineID": [1, 1, 2],
        "event": ["E1", "E2", "E1"]
    })

    result = pivot_table_generic(df, ["machineID"], "event", "suffix")

    assert "E1_suffix" in result.columns
    assert "E2_suffix" in result.columns
    assert result.loc[result["machineID"] == 1, "E1_suffix"].values[0] == 1
    assert result.loc[result["machineID"] == 2, "E1_suffix"].values[0] == 1
    assert result.loc[result["machineID"] == 2, "E2_suffix"].values[0] == 0

def test_add_time_features() -> None:
    df = pd.DataFrame({
        "datetime": pd.to_datetime(["2015-01-01 12:00:00"])
    })

    result = add_time_features(df)

    assert "hour" in result.columns
    assert result["hour"].values[0] == 12
    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns

def test_add_global_flags() -> None:
    df = pd.DataFrame({
        "comp1_error": [1, 0],
        "comp2_error": [0, 0],
        "comp1_maint": [0, 1],
        "comp1_fail": [0, 0]
    })

    result = add_global_flags(df)

    assert result["any_error"].iloc[0] == 1
    assert result["any_error"].iloc[1] == 0
    assert result["any_maint"].iloc[1] == 1
    assert result["any_fail"].iloc[0] == 0
