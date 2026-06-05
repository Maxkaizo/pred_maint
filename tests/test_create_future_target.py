import pandas as pd

from app.tasks.create_future_target import create_future_target


def test_create_future_target() -> None:
    # Machine 1 fails at T=10
    # Gap=1, Horizon=2
    # Failure at T=10 should mark [10-2-1, 10-1] = [7, 9] as 1
    df = pd.DataFrame({
        "datetime": pd.to_datetime([f"2015-01-01 {h:02d}:00:00" for h in range(12)]),
        "machineID": [1] * 12,
        "any_fail": [0] * 12
    })
    df.loc[10, "any_fail"] = 1

    result = create_future_target(df, gap="1h", horizon="2h")

    assert "any_fail_future" in result.columns
    # T=7, 8 should be 1
    assert result.loc[7, "any_fail_future"] == 1
    assert result.loc[8, "any_fail_future"] == 1
    # Others should be 0
    assert result.loc[6, "any_fail_future"] == 0
    assert result.loc[9, "any_fail_future"] == 0
    assert result.loc[10, "any_fail_future"] == 0
