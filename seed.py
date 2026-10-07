"""
Takes real rows from SupervisedLandslideDataset.csv, sends them to the API (/predict),
and compares the prediction with the true label. Every result is saved in Neon.

Steps:
    1. Start the API:      uvicorn main:app --reload      (leave this terminal running)
    2. In a 2nd terminal:  python seed.py
"""
import os

import pandas as pd
import requests

# ---------------- SETTINGS (change these) ----------------
CSV_PATH = os.getenv(
    "CSV_PATH",
    r"C:\Users\VIVEK GUPTA\OneDrive\Desktop\landslide model\SupervisedLandslideDataset.csv",
)
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
ROWS_PER_CLASS = 5      # 5 landslide rows + 5 non-landslide rows = 10 requests
RANDOM_STATE = 7        # change it to get different rows
# ----------------------------------------------------------

DAYS = 36
WEATHER_VARS = ["precip", "temp", "humidity", "wind", "air"]


def get_extra_columns(df):
    """Other raw CSV columns the model was trained on (asked from the API)."""
    r = requests.get(f"{BASE_URL}/model/info", timeout=30)
    if r.status_code != 200:
        raise SystemExit(f"/model/info returned {r.status_code}: {r.text}")
    trained_on = set(r.json()["feature_columns"])
    skip = {"date", "slope", "lat", "lon", "label"}
    weather_cols = {f"{v}{i}" for v in WEATHER_VARS for i in range(DAYS)}
    return [c for c in df.columns if c in trained_on and c not in skip and c not in weather_cols]


def number_or_none(value):
    value = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(value) else float(value)


def row_to_payload(idx, row, extra_cols):
    """Convert one CSV row to the JSON the /predict route expects.
    Returns (payload, None) if usable, or (None, reason) if not."""
    slope = number_or_none(row.get("slope"))
    if slope is None:
        return None, "slope missing"

    payload = {"slope": slope, "location_name": f"CSV row {idx}"}

    date = pd.to_datetime(row.get("date"), errors="coerce")
    if not pd.isna(date):  # date is optional, the model does not use it
        payload["date"] = date.strftime("%Y-%m-%d")
    for name in ("lat", "lon"):
        value = number_or_none(row.get(name))
        if value is not None:
            payload[name] = value

    for var in WEATHER_VARS:
        series = pd.to_numeric(
            pd.Series([row.get(f"{var}{i}") for i in range(DAYS)]), errors="coerce"
        )
        if series.isna().all():
            return None, f"all {var} values missing"
        series = series.fillna(series.mean())  # fill small gaps with this row's own average
        payload[var] = [float(x) for x in series]

    extra = {}
    for col in extra_cols:
        value = number_or_none(row.get(col))
        if value is not None:
            extra[col] = value
    payload["extra_features"] = extra
    return payload, None


def collect_usable_rows(df, extra_cols):
    """For each class (1 = landslide, 0 = no landslide) keep drawing random rows
    until we have ROWS_PER_CLASS usable ones."""
    chosen = []
    skip_reasons = {}
    for label_value in (1, 0):
        group = df[df["label"] == label_value].sample(frac=1, random_state=RANDOM_STATE)
        found = 0
        for idx, row in group.iterrows():
            payload, reason = row_to_payload(idx, row, extra_cols)
            if payload is None:
                key = f"label={label_value}: {reason}"
                skip_reasons[key] = skip_reasons.get(key, 0) + 1
                continue
            chosen.append((idx, label_value, payload))
            found += 1
            if found >= ROWS_PER_CLASS:
                break
        if found < ROWS_PER_CLASS:
            print(f"WARNING: only found {found} usable rows with label={label_value}")
    return chosen, skip_reasons


def main():
    try:
        requests.get(f"{BASE_URL}/health", timeout=10)
    except requests.exceptions.ConnectionError:
        print("Could not connect. Start the API first in another terminal: uvicorn main:app --reload")
        return

    df = pd.read_csv(CSV_PATH, low_memory=False)
    extra_cols = get_extra_columns(df)
    print(f"Extra columns sent as extra_features: {extra_cols if extra_cols else 'none'}")

    chosen, skip_reasons = collect_usable_rows(df, extra_cols)
    if skip_reasons:
        print("\nRows skipped while searching (reason: count):")
        for reason, count in skip_reasons.items():
            print(f"  {reason}: {count}")
    print()

    correct = total = 0
    for idx, actual, payload in chosen:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=60)
        if r.status_code != 200:
            print(f"Row {idx}: FAILED {r.status_code} {r.text[:300]}")
            continue

        d = r.json()
        predicted = int(d["landslide_predicted"])
        total += 1
        correct += int(actual == predicted)
        mark = "OK " if actual == predicted else "BAD"
        print(f"[{mark}] id={d['id']:<4} row={idx:<6} actual={actual} predicted={predicted} "
              f"prob={d['probability']:.3f} risk={d['risk_level']}")

    if total:
        print(f"\nMatched true label on {correct}/{total} rows")
    print("\n/stats:", requests.get(f"{BASE_URL}/stats", timeout=30).json())


if __name__ == "__main__":
    main()
