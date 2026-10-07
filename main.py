"""
Landslide Prediction API  (FastAPI + Neon PostgreSQL)

Run:   uvicorn main:app --reload
Docs:  http://127.0.0.1:8000/docs

Files needed in this folder:
    main.py, features.py, landslide_model.joblib (made by train_model.py), .env

.env must contain:
    DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
"""
import os
import warnings
from contextlib import asynccontextmanager
from datetime import date as date_type
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import sklearn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from features import DAYS, engineer_features

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

# ----------------------------------------------------------------------------
# 1. CONFIG + DATABASE
# ----------------------------------------------------------------------------
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "landslide_model.joblib"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))  # probability >= this => landslide

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing. Put it in a .env file.")

# pool_pre_ping: Neon pauses idle databases, this reconnects automatically
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """One row per prediction made through the API."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    location_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    input_data: Mapped[dict] = mapped_column(JSON)  # raw input we received
    probability: Mapped[float] = mapped_column(Float)
    prediction: Mapped[int] = mapped_column(Integer)  # 1 = landslide, 0 = no landslide
    risk_level: Mapped[str] = mapped_column(String(10), index=True)  # Low / Medium / High


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ----------------------------------------------------------------------------
# 2. SCHEMAS (Pydantic v2)
# ----------------------------------------------------------------------------
class LandslideInput(BaseModel):
    date: Optional[date_type] = Field(
        default=None, description="Optional (YYYY-MM-DD). Only for your records, the model does not use it."
    )
    slope: float = Field(ge=0, description="Slope value, same unit as your dataset")
    lat: Optional[float] = Field(default=None, ge=-90, le=90, description="Latitude of the location")
    lon: Optional[float] = Field(default=None, ge=-180, le=180, description="Longitude of the location")
    precip: list[float] = Field(description="36 daily rainfall values (index 0 = most recent day)")
    temp: list[float] = Field(description="36 daily temperature values")
    humidity: list[float] = Field(description="36 daily humidity values")
    wind: list[float] = Field(description="36 daily wind values")
    air: list[float] = Field(description="36 daily air values")
    location_name: Optional[str] = Field(default=None, max_length=100)
    extra_features: dict[str, float] = Field(
        default_factory=dict,
        description="Any other raw dataset columns the model was trained on (optional)",
    )

    @field_validator("precip", "temp", "humidity", "wind", "air")
    @classmethod
    def must_have_36_values(cls, v: list[float]) -> list[float]:
        if len(v) != DAYS:
            raise ValueError(f"must contain exactly {DAYS} daily values, got {len(v)}")
        return v


class PredictionOut(BaseModel):
    id: int
    created_at: datetime
    location_name: Optional[str]
    landslide_predicted: bool
    probability: float
    risk_level: str
    key_factors: dict[str, float]


class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    location_name: Optional[str]
    probability: float
    prediction: int
    risk_level: str


class HistoryDetail(HistoryItem):
    input_data: dict


# ----------------------------------------------------------------------------
# 3. MODEL + PREDICTION HELPERS
# ----------------------------------------------------------------------------
state: dict = {"artifact": None}


def get_artifact() -> dict:
    if state["artifact"] is None:
        raise HTTPException(
            503,
            f"Model not loaded. Run train_model.py so '{MODEL_PATH}' exists, then restart the API. "
            "Check the uvicorn terminal for the reason.",
        )
    return state["artifact"]


def input_to_frame(item: LandslideInput) -> pd.DataFrame:
    """Turn one API input into a one-row DataFrame with raw dataset columns."""
    row = {**item.extra_features, "slope": item.slope}
    if item.lat is not None:
        row["lat"] = item.lat
    if item.lon is not None:
        row["lon"] = item.lon
    for var in ("precip", "temp", "humidity", "wind", "air"):
        for i, value in enumerate(getattr(item, var)):
            row[f"{var}{i}"] = value
    return pd.DataFrame([row])


def risk_level(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    if prob < 0.60:
        return "Medium"
    return "High"


KEY_FACTOR_NAMES = [
    "rain_1d", "rain_3d", "rain_7d", "rain_30d",
    "antecedent_rainfall_index", "heavy_rain_days_7d", "slope",
]


def predict_and_save(items: list[LandslideInput], db: Session) -> list[Prediction]:
    art = get_artifact()
    try:
        raw = pd.concat([input_to_frame(i) for i in items], ignore_index=True)
        feats = engineer_features(raw)
        X = feats.reindex(columns=art["feature_columns"])  # same columns/order as training
        probs = art["model"].predict_proba(X)[:, 1]
    except Exception as e:  # keep the message readable for the demo
        raise HTTPException(500, f"Prediction failed: {e}")

    rows = []
    for item, prob, (_, frow) in zip(items, probs, feats.iterrows()):
        prob = float(prob)
        data = item.model_dump(mode="json")
        data["key_factors"] = {k: round(float(frow[k]), 3) for k in KEY_FACTOR_NAMES if k in frow}
        row = Prediction(
            location_name=item.location_name,
            input_data=data,
            probability=round(prob, 4),
            prediction=int(prob >= THRESHOLD),
            risk_level=risk_level(prob),
        )
        db.add(row)
        rows.append(row)
    db.commit()
    for r in rows:
        db.refresh(r)
    return rows


def to_output(row: Prediction) -> PredictionOut:
    return PredictionOut(
        id=row.id,
        created_at=row.created_at,
        location_name=row.location_name,
        landslide_predicted=bool(row.prediction),
        probability=row.probability,
        risk_level=row.risk_level,
        key_factors=row.input_data.get("key_factors", {}),
    )


# ----------------------------------------------------------------------------
# 4. APP
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)  # creates the table in Neon if missing

    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: {MODEL_PATH} not found. Run train_model.py. /predict will return 503.")
    else:
        try:
            art = joblib.load(MODEL_PATH)
            if not isinstance(art, dict) or "model" not in art or "feature_columns" not in art:
                raise ValueError("file is not in the expected format. Re-run train_model.py")
            saved_version = art.get("sklearn_version")
            if saved_version and saved_version != sklearn.__version__:
                print(
                    f"WARNING: model trained with scikit-learn {saved_version} but the API runs "
                    f"{sklearn.__version__}. Re-run train_model.py with this same Python."
                )
            state["artifact"] = art
            print(f"Model loaded: {art.get('name', 'model')} with {len(art['feature_columns'])} features")
        except Exception as e:
            print(f"ERROR loading model: {e}")
    yield


app = FastAPI(
    title="Landslide Prediction API",
    description="Send weather + terrain data, get landslide probability. Predictions are stored in Neon PostgreSQL.",
    version="1.1.0",
    lifespan=lifespan,
)

# Allow a frontend (React, etc.) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# 5. ROUTES
# ----------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    return {"message": "Landslide Prediction API is running", "docs": "/docs"}


@app.get("/health", tags=["General"])
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "status": "ok" if (db_ok and state["artifact"] is not None) else "degraded",
        "database_connected": db_ok,
        "model_loaded": state["artifact"] is not None,
    }


@app.get("/model/info", tags=["General"])
def model_info():
    art = get_artifact()
    return {
        "model_name": art.get("name", "unknown"),
        "number_of_features": len(art["feature_columns"]),
        "test_metrics": art.get("metrics"),
        "uses_lat_lon": art.get("uses_lat_lon"),
        "sklearn_trained_with": art.get("sklearn_version"),
        "sklearn_running": sklearn.__version__,
        "threshold": THRESHOLD,
        "risk_levels": {"Low": "< 0.30", "Medium": "0.30 - 0.60", "High": ">= 0.60"},
        "required_input": "slope, lat, lon, and 36 daily values each for precip, temp, humidity, wind, air",
        "feature_columns": art["feature_columns"],
    }


@app.get("/sample-input", tags=["General"])
def sample_input(scenario: str = Query("heavy_rain", pattern="^(heavy_rain|dry)$")):
    """Ready-made valid input. Copy it into POST /predict to test quickly."""
    if scenario == "heavy_rain":
        precip = [60, 45, 50, 30, 25, 20, 15] + [10.0] * 29
        return {
            "slope": 35.0, "lat": 24.8, "lon": 93.9, "location_name": "Sample hill (heavy rain)",
            "precip": precip, "temp": [22.0] * DAYS, "humidity": [90.0] * DAYS,
            "wind": [6.0] * DAYS, "air": [20.0] * DAYS, "extra_features": {},
        }
    return {
        "slope": 8.0, "lat": 23.8, "lon": 91.3, "location_name": "Sample plain (dry)",
        "precip": [0.0] * DAYS, "temp": [15.0] * DAYS, "humidity": [45.0] * DAYS,
        "wind": [3.0] * DAYS, "air": [14.0] * DAYS, "extra_features": {},
    }


@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
def predict(item: LandslideInput, db: Session = Depends(get_db)):
    """Predict landslide for one location. The result is saved to history."""
    return to_output(predict_and_save([item], db)[0])


@app.post("/predict/batch", response_model=list[PredictionOut], tags=["Prediction"])
def predict_batch(items: list[LandslideInput], db: Session = Depends(get_db)):
    """Predict for many locations at once (max 50)."""
    if not 1 <= len(items) <= 50:
        raise HTTPException(422, "Send between 1 and 50 items.")
    return [to_output(r) for r in predict_and_save(items, db)]


@app.get("/history", response_model=list[HistoryItem], tags=["History"])
def get_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    risk_level: Optional[str] = Query(None, pattern="^(Low|Medium|High)$"),
    db: Session = Depends(get_db),
):
    """Latest predictions first. Optional filter: ?risk_level=High"""
    query = select(Prediction).order_by(Prediction.id.desc()).limit(limit).offset(offset)
    if risk_level:
        query = query.where(Prediction.risk_level == risk_level)
    return db.scalars(query).all()


@app.get("/history/latest", response_model=HistoryDetail, tags=["History"])
def get_latest_prediction(db: Session = Depends(get_db)):
    """The most recent prediction, with its full input."""
    row = db.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(1)).first()
    if not row:
        raise HTTPException(404, "No predictions yet")
    return row

@app.get("/history/{prediction_id}", response_model=HistoryDetail, tags=["History"])
def get_history_item(prediction_id: int, db: Session = Depends(get_db)):
    row = db.get(Prediction, prediction_id)
    if not row:
        raise HTTPException(404, "Prediction not found")
    return row


@app.delete("/history/{prediction_id}", tags=["History"])
def delete_history_item(prediction_id: int, db: Session = Depends(get_db)):
    row = db.get(Prediction, prediction_id)
    if not row:
        raise HTTPException(404, "Prediction not found")
    db.delete(row)
    db.commit()
    return {"deleted": prediction_id}


@app.get("/stats", tags=["History"])
def stats(db: Session = Depends(get_db)):
    """Summary numbers, handy for a dashboard."""
    total = db.scalar(select(func.count(Prediction.id))) or 0
    avg = db.scalar(select(func.avg(Prediction.probability)))
    landslides = db.scalar(select(func.count(Prediction.id)).where(Prediction.prediction == 1)) or 0
    by_risk = dict(db.execute(select(Prediction.risk_level, func.count()).group_by(Prediction.risk_level)).all())
    return {
        "total_predictions": total,
        "landslides_predicted": landslides,
        "average_probability": round(float(avg), 4) if avg is not None else None,
        "by_risk_level": {k: by_risk.get(k, 0) for k in ("Low", "Medium", "High")},
    }
