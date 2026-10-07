"""
Trains the landslide model and saves ONE file for the API: landslide_model.joblib

Run it with the SAME Python that runs the API (this avoids scikit-learn version errors):
    python train_model.py

Keep features.py in the same folder.
"""
import os
import sys

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features import engineer_features

# ---------------- SETTINGS ----------------
CSV_PATH = os.getenv(
    "CSV_PATH",
    r"C:\Users\VIVEK GUPTA\OneDrive\Desktop\landslide model\SupervisedLandslideDataset.csv",
)
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "landslide_model.joblib")
USE_LAT_LON = True          # set False to train without location (compare the scores!)
MISSING_THRESHOLD = 0.50    # drop columns with more than 50% missing values
# Columns that are only known AFTER a landslide happened, or are just IDs/text
DROP_COLUMNS = [
    "Unnamed: 0", "id", "country", "fatalities", "injuries",
    "type", "trigger", "severity", "location",
]
# ------------------------------------------


def check_missing_shortcuts(df: pd.DataFrame) -> None:
    """Warn if a column is missing much more often in one class than the other.
    The model can learn 'is it missing?' instead of learning real landslide patterns."""
    miss = df.drop(columns="label").isna().groupby(df["label"]).mean().T
    miss["gap"] = miss.max(axis=1) - miss.min(axis=1)
    suspicious = miss[miss["gap"] > 0.30].sort_values("gap", ascending=False).head(10)
    if suspicious.empty:
        print("Missing-value check: OK, no column is missing much more in one class.")
    else:
        print("WARNING - columns missing much more often in one class (possible shortcut):")
        print(suspicious.round(2).to_string())
        print("(date is dropped from the model on purpose. Check the others.)\n")


def main():
    print(f"Python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    print(f"Interpreter: {sys.executable}\n")

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    print("Label distribution:")
    print(df["label"].value_counts().to_string(), "\n")

    check_missing_shortcuts(df)

    df = engineer_features(df)

    # drop columns with too many missing values
    missing_ratio = df.isnull().mean()
    too_empty = missing_ratio[missing_ratio > MISSING_THRESHOLD].index.drop("label", errors="ignore")
    df = df.drop(columns=too_empty)
    print(f"Dropped for >{MISSING_THRESHOLD:.0%} missing: {list(too_empty)}")

    if not USE_LAT_LON:
        df = df.drop(columns=["lat", "lon"], errors="ignore")

    y = df["label"]
    X = df.drop(columns=["label"])
    numeric = X.select_dtypes(include="number").columns.tolist()
    skipped = [c for c in X.columns if c not in numeric]
    if skipped:
        print(f"Non-numeric columns not used: {skipped}")
    X = X[numeric]
    print(f"Training with {X.shape[1]} features on {X.shape[0]} rows\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # One Pipeline: fills missing values, then Random Forest.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", RandomForestClassifier(
            n_estimators=300, min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
    }
    print("Test metrics:", metrics)
    print("\n", classification_report(y_test, pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    # Which features matter most? One feature far ahead of the rest can mean a shortcut.
    importance = pd.Series(model.named_steps["model"].feature_importances_, index=X.columns)
    print("\nTop 15 features:")
    print(importance.sort_values(ascending=False).head(15).round(4).to_string())

    artifact = {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "name": "RandomForest",
        "sklearn_version": sklearn.__version__,
        "metrics": metrics,
        "uses_lat_lon": USE_LAT_LON,
    }
    joblib.dump(artifact, OUTPUT_PATH, compress=3)
    print(f"\nSaved {OUTPUT_PATH} ({X.shape[1]} features). Restart the API now.")


if __name__ == "__main__":
    main()
