"""
SIH26001 - Landslide Risk Monitoring backend.

Run locally:
    pip install fastapi uvicorn joblib pandas scikit-learn python-multipart twilio
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /                       -> health check
    GET  /zones                  -> current risk score for all demo zones
    POST /predict                -> risk prediction for custom features
    POST /citizen-report        -> submit a geo-tagged report
    GET  /citizen-reports       -> list submitted reports
    POST /alert/trigger         -> manually trigger an alert

Alerts:
    Set TWILIO_SID / TWILIO_TOKEN / TWILIO_FROM / ALERT_TO
    to send real SMS via Twilio.
    Without them, alerts are logged in mock mode.
"""

import os
import json
import random
from datetime import datetime
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(title="SIH26001 - Landslide Risk Monitoring API")


# -------------------------------------------------------------------
# CORS
# -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# FILE PATHS
# Works whether the project is run from the backend folder or repo root.
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "models", "landslide_risk_model.joblib"),
    os.path.join(BASE_DIR, "..", "models", "landslide_risk_model.joblib"),
]

MODEL_PATH = next(
    (os.path.abspath(path) for path in MODEL_CANDIDATES if os.path.exists(path)),
    None,
)

ALERT_LOG = os.path.join(BASE_DIR, "alert_log.json")
REPORTS_LOG = os.path.join(BASE_DIR, "citizen_reports.json")

RISK_ALERT_THRESHOLD = 0.6


# -------------------------------------------------------------------
# Load ML model safely
# -------------------------------------------------------------------
model = None
FEATURES = []

if MODEL_PATH:
    try:
        bundle = joblib.load(MODEL_PATH)

        # Expected format:
        # {"model": trained_model, "features": [...]}
        if isinstance(bundle, dict):
            model = bundle.get("model")
            FEATURES = bundle.get("features", [])
        else:
            # Extra fallback if a raw sklearn model was saved.
            model = bundle

        if not FEATURES:
            # Standard feature order used by this project.
            FEATURES = [
                "rainfall_mm_24h",
                "rainfall_mm_7d",
                "slope_deg",
                "elevation_m",
                "soil_moisture",
                "historical_landslide_density",
                "distance_to_road_km",
                "ndvi_vegetation_index",
            ]

        print(f"[MODEL] Loaded successfully: {MODEL_PATH}")
        print(f"[MODEL] Features: {FEATURES}")

    except Exception as e:
        print(f"[MODEL ERROR] Could not load model: {type(e).__name__}: {e}")
else:
    print("[MODEL ERROR] Model file not found.")
    print("[MODEL] Checked:")
    for path in MODEL_CANDIDATES:
        print(f"  - {os.path.abspath(path)}")


# -------------------------------------------------------------------
# Demo zones
# -------------------------------------------------------------------
DEMO_ZONES = [
    {"id": "z1", "name": "East Khasi Hills", "lat": 25.57, "lon": 91.88},
    {"id": "z2", "name": "West Garo Hills", "lat": 25.51, "lon": 90.21},
    {"id": "z3", "name": "Kohima", "lat": 25.67, "lon": 94.11},
    {"id": "z4", "name": "Dima Hasao", "lat": 25.03, "lon": 93.02},
    {"id": "z5", "name": "Upper Subansiri", "lat": 27.98, "lon": 93.98},
    {"id": "z6", "name": "West Kameng", "lat": 27.22, "lon": 92.35},
    {"id": "z7", "name": "Aizawl", "lat": 23.73, "lon": 92.72},
    {"id": "z8", "name": "Churachandpur", "lat": 24.33, "lon": 93.68},
]


# -------------------------------------------------------------------
# Request models
# -------------------------------------------------------------------
class PredictInput(BaseModel):
    rainfall_mm_24h: float
    rainfall_mm_7d: float
    slope_deg: float
    elevation_m: float
    soil_moisture: float
    historical_landslide_density: float
    distance_to_road_km: float
    ndvi_vegetation_index: float


class AlertRequest(BaseModel):
    zone_id: str
    risk_score: float
    message: Optional[str] = None


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _random_live_features():
    """
    Demo replacement for live IMD/SRTM/SMAP data.
    Replace with real API/grid data when available.
    """
    return {
        "rainfall_mm_24h": round(random.uniform(0, 180), 1),
        "rainfall_mm_7d": round(random.uniform(0, 600), 1),
        "slope_deg": round(random.uniform(5, 55), 1),
        "elevation_m": round(random.uniform(200, 2500), 0),
        "soil_moisture": round(random.uniform(0.1, 0.95), 2),
        "historical_landslide_density": round(random.uniform(0, 3), 2),
        "distance_to_road_km": round(random.uniform(0, 10), 2),
        "ndvi_vegetation_index": round(random.uniform(0.2, 0.9), 2),
    }


def _fallback_risk(features: dict) -> float:
    """
    Demo fallback so the dashboard does not crash if the ML model
    is temporarily unavailable or has an incompatible feature format.
    """
    score = (
        (features.get("rainfall_mm_24h", 0) / 180.0) * 0.30
        + (features.get("rainfall_mm_7d", 0) / 600.0) * 0.20
        + (features.get("slope_deg", 0) / 55.0) * 0.20
        + features.get("soil_moisture", 0) * 0.15
        + (features.get("historical_landslide_density", 0) / 3.0) * 0.15
    )

    return round(max(0.0, min(0.99, score)), 3)


def _predict(features: dict) -> float:
    """
    Run the trained model.
    If the model cannot predict, use a safe demo fallback instead of
    returning HTTP 500 from /zones.
    """
    if model is None:
        print("[PREDICTION] Model unavailable -> using demo fallback.")
        return _fallback_risk(features)

    try:
        # Make sure every expected model feature exists.
        missing = [f for f in FEATURES if f not in features]

        if missing:
            raise ValueError(f"Missing model features: {missing}")

        # Preserve the exact feature order expected by the trained model.
        row = pd.DataFrame(
            [[features[f] for f in FEATURES]],
            columns=FEATURES,
        )

        # sklearn classifiers normally expose predict_proba().
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]

            # Find class 1 when classes_ is available.
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    class_index = classes.index(1)
                else:
                    class_index = len(probabilities) - 1
            else:
                class_index = min(1, len(probabilities) - 1)

            return float(probabilities[class_index])

        # Fallback for models that only expose predict().
        prediction = model.predict(row)[0]
        return float(prediction)

    except Exception as e:
        print(f"[PREDICTION ERROR] {type(e).__name__}: {e}")
        print(f"[PREDICTION ERROR] FEATURES = {FEATURES}")
        print(f"[PREDICTION ERROR] INPUT = {features}")
        print("[PREDICTION] Using demo fallback score.")
        return _fallback_risk(features)


def _risk_level(risk: float) -> str:
    if risk >= RISK_ALERT_THRESHOLD:
        return "high"
    if risk >= 0.35:
        return "moderate"
    return "low"


def _log_json(path: str, entry: dict):
    data = []

    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                data = []

    except Exception as e:
        print(f"[JSON LOG WARNING] Could not read {path}: {e}")
        data = []

    data.append(entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _send_alert(zone_name: str, risk_score: float, message: str):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "zone": zone_name,
        "risk_score": float(risk_score),
        "message": message,
    }

    sid = os.getenv("TWILIO_SID")
    token = os.getenv("TWILIO_TOKEN")
    from_no = os.getenv("TWILIO_FROM")
    to_no = os.getenv("ALERT_TO")

    if sid and token and from_no and to_no:
        try:
            from twilio.rest import Client

            client = Client(sid, token)

            client.messages.create(
                body=message,
                from_=from_no,
                to=to_no,
            )

            entry["mode"] = "twilio_sms_sent"

        except Exception as e:
            entry["mode"] = "twilio_error_mock_logged"
            entry["error"] = str(e)
            print(f"[TWILIO ERROR] {e}")
            print(f"[ALERT-MOCK] {message}")

    else:
        entry["mode"] = "mock_logged_only"
        print(f"[ALERT-MOCK] {message}")

    _log_json(ALERT_LOG, entry)
    return entry


# -------------------------------------------------------------------
# API endpoints
# -------------------------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "SIH26001 landslide risk API",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "features": FEATURES,
    }


@app.get("/zones")
def get_zones():
    """
    Generate demo live-like features for every zone, calculate risk,
    and auto-trigger alerts when risk >= 0.6.
    """
    results = []

    for z in DEMO_ZONES:
        try:
            feats = _random_live_features()
            risk = _predict(feats)

            entry = {
                **z,
                "risk_score": round(risk, 3),
                "features": feats,
                "risk_level": _risk_level(risk),
            }

            if risk >= RISK_ALERT_THRESHOLD:
                _send_alert(
                    z["name"],
                    risk,
                    (
                        f"HIGH LANDSLIDE RISK in {z['name']} "
                        f"(score {risk:.2f}). Alert district admin."
                    ),
                )

            results.append(entry)

        except Exception as e:
            # One bad zone must not break the entire dashboard.
            print(f"[ZONE ERROR] {z['name']}: {type(e).__name__}: {e}")

            fallback_features = _random_live_features()
            fallback_risk = _fallback_risk(fallback_features)

            results.append({
                **z,
                "risk_score": fallback_risk,
                "features": fallback_features,
                "risk_level": _risk_level(fallback_risk),
                "warning": "Demo fallback score used.",
            })

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "zones": results,
    }


@app.post("/predict")
def predict(inp: PredictInput):
    features = inp.dict()
    risk = _predict(features)

    return {
        "risk_score": round(risk, 3),
        "risk_level": _risk_level(risk),
    }


@app.post("/citizen-report")
async def citizen_report(
    lat: float = Form(...),
    lon: float = Form(...),
    description: str = Form(...),
    photo: Optional[UploadFile] = File(None),
):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "lat": lat,
        "lon": lon,
        "description": description,
        "photo_filename": photo.filename if photo else None,
    }

    _log_json(REPORTS_LOG, entry)

    return {
        "status": "received",
        "report": entry,
    }


@app.get("/citizen-reports")
def list_reports():
    if not os.path.exists(REPORTS_LOG):
        return {"reports": []}

    try:
        with open(REPORTS_LOG, "r", encoding="utf-8") as f:
            reports = json.load(f)

        return {"reports": reports}

    except Exception as e:
        print(f"[REPORTS ERROR] {e}")
        return {"reports": []}


@app.post("/alert/trigger")
def trigger_alert(req: AlertRequest):
    msg = req.message or (
        f"Landslide risk alert for zone "
        f"{req.zone_id} (score {req.risk_score:.2f})"
    )

    return _send_alert(
        req.zone_id,
        req.risk_score,
        msg,
    )
