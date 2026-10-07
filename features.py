"""
Feature engineering, shared by train_model.py (training) and main.py (API).

Both files import this ONE function, so the API always builds exactly the same
features the model was trained on.

NOTE: date features (year, month, day...) are deliberately NOT created.
In this dataset the non-landslide rows have no date, so date features would let the
model "cheat" (date present = landslide) instead of learning from rainfall and terrain.
"""
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)  # harmless, from adding many columns

DAYS = 36  # 36 days of weather: index 0 = event day, 35 = oldest
WINDOWS = [3, 7, 14, 30, 36]


def _cols(var: str, n: int) -> list[str]:
    return [f"{var}{i}" for i in range(n)]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns (slope, precip0..35, temp0..35, ...) -> raw + engineered columns."""
    df = df.copy()
    precip_all = _cols("precip", DAYS)

    # ---- rainfall totals ----
    df["rain_1d"] = df["precip0"]
    for n in (3, 7, 30):
        df[f"rain_{n}d"] = df[_cols("precip", n)].sum(axis=1)
    df["rain_35d"] = df[precip_all].sum(axis=1)

    # ---- rainfall stats over all 36 days ----
    df["rain_mean_35d"] = df[precip_all].mean(axis=1)
    df["rain_max_35d"] = df[precip_all].max(axis=1)
    df["rain_std_35d"] = df[precip_all].std(axis=1)
    df["rain_min_35d"] = df[precip_all].min(axis=1)

    # ---- rainfall stats over shorter windows ----
    for w in (3, 7, 14, 30):
        block = df[_cols("precip", w)]
        df[f"rain_max_{w}d"] = block.max(axis=1)
        df[f"rain_mean_{w}d"] = block.mean(axis=1)
        df[f"rain_std_{w}d"] = block.std(axis=1)

    # ---- wet / heavy rain day counts ----
    for w in WINDOWS:
        df[f"wet_days_{w}d"] = (df[_cols("precip", w)] > 5).sum(axis=1)
    for w in (7, 14, 30):
        df[f"heavy_rain_days_{w}d"] = (df[_cols("precip", w)] > 20).sum(axis=1)

    # ---- rain trend ----
    df["recent_rain_7d"] = df[_cols("precip", 7)].sum(axis=1)
    df["previous_rain_7d"] = df[[f"precip{i}" for i in range(7, 14)]].sum(axis=1)
    df["rain_trend_7d"] = df["recent_rain_7d"] - df["previous_rain_7d"]
    df["rain_ratio_7d"] = df["recent_rain_7d"] / (df["previous_rain_7d"] + 1e-6)

    # ---- antecedent rainfall index (recent days weigh more) ----
    weights = np.exp(-np.arange(DAYS) / 7)
    weights = weights / weights.sum()
    df["antecedent_rainfall_index"] = df[precip_all].fillna(0).values @ weights

    # ---- temp / humidity / wind / air stats ----
    stats_per_var = {
        "temp": ("mean", "max", "min", "std"),
        "humidity": ("mean", "max", "min", "std"),
        "wind": ("mean", "max", "std"),
        "air": ("mean", "max", "min", "std"),
    }
    for var, stats in stats_per_var.items():
        for w in WINDOWS:
            block = df[_cols(var, w)]
            for s in stats:
                df[f"{var}_{s}_{w}d"] = getattr(block, s)(axis=1)

    # ---- terrain ----
    df["slope_squared"] = df["slope"] ** 2

    return df.drop(columns=["date"], errors="ignore")
