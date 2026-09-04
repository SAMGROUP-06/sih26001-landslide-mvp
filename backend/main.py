"""
SIH26001 - Landslide Risk Monitoring backend.

Run locally:
    pip install fastapi uvicorn joblib pandas scikit-learn python-multipart twilio
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /                       -> health check
    GET  /zones                  -> current risk score for all demo zones (feeds the GIS dashboard)
    POST /predict                -> risk prediction for a custom set of features
    POST /citizen-report         -> submit a geo-tagged crack/slope-movement report
    GET  /citizen-reports        -> list submitted reports
    POST /alert/trigger          -> manually trigger an alert for a zone (also auto-triggers when a zone crosses threshold)

Alerts: set TWILIO_SID / TWILIO_TOKEN / TWILIO_FROM / ALERT_TO env vars to send
real SMS via Twilio. Without them, alerts are logged to console/alert_log.json
(mock mode) so the pipeline still runs end-to-end for a demo.
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) MODEL_PATH = os.path.join(BASE_DIR, "models", "landslide_risk_model.joblib")
bundle = joblib.load(MODEL_PATH)
model, FEATURES = bundle["model"], bundle["features"]

ALERT_LOG = "alert_log.json"
REPORTS_LOG = "citizen_reports.json"
RISK_ALERT_THRESHOLD = 0.6  # probability above which an auto-alert fires

# ---- Demo zones (replace lat/lon + live feature values with real district
# centroids + live IMD/SRTM/SMAP pulls once you wire up real data sources) ----
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


def _random_live_features():
    """Stand-in for a live IMD/SRTM/SMAP data pull for a zone.
    Replace this with real API calls / cached grid lookups."""
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


def _predict(features: dict) -> float:
    row = pd.DataFrame([[features[f] for f in FEATURES]], columns=FEATURES)
    return float(model.predict_proba(row)[0][1])


def _log_json(path: str, entry: dict):
    data = []
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data.append(entry)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _send_alert(zone_name: str, risk_score: float, message: str):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "zone": zone_name,
        "risk_score": risk_score,
        "message": message,
    }
    sid, token, from_no, to_no = (
        os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"),
        os.getenv("TWILIO_FROM"), os.getenv("ALERT_TO"),
    )
    if sid and token and from_no and to_no:
        from twilio.rest import Client
        client = Client(sid, token)
        client.messages.create(body=message, from_=from_no, to=to_no)
        entry["mode"] = "twilio_sms_sent"
    else:
        entry["mode"] = "mock_logged_only (set TWILIO_* env vars for real SMS)"
        print(f"[ALERT-MOCK] {message}")
    _log_json(ALERT_LOG, entry)
    return entry


@app.get("/")
def health():
    return {"status": "ok", "service": "SIH26001 landslide risk API"}


@app.get("/zones")
def get_zones():
    """Pull live-ish features per zone, score them, auto-alert if above threshold."""
    results = []
    for z in DEMO_ZONES:
        feats = _random_live_features()
        risk = _predict(feats)
        entry = {**z, "risk_score": round(risk, 3), "features": feats,
                  "risk_level": "high" if risk >= RISK_ALERT_THRESHOLD else ("moderate" if risk >= 0.35 else "low")}
        if risk >= RISK_ALERT_THRESHOLD:
            _send_alert(z["name"], risk,
                        f"HIGH LANDSLIDE RISK in {z['name']} (score {risk:.2f}). Alert district admin.")
        results.append(entry)
    return {"generated_at": datetime.utcnow().isoformat(), "zones": results}


@app.post("/predict")
def predict(inp: PredictInput):
    risk = _predict(inp.dict())
    return {"risk_score": round(risk, 3),
            "risk_level": "high" if risk >= RISK_ALERT_THRESHOLD else ("moderate" if risk >= 0.35 else "low")}


@app.post("/citizen-report")
async def citizen_report(
    lat: float = Form(...), lon: float = Form(...),
    description: str = Form(...), photo: Optional[UploadFile] = File(None),
):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "lat": lat, "lon": lon, "description": description,
        "photo_filename": photo.filename if photo else None,
    }
    _log_json(REPORTS_LOG, entry)
    return {"status": "received", "report": entry}


@app.get("/citizen-reports")
def list_reports():
    if not os.path.exists(REPORTS_LOG):
        return {"reports": []}
    with open(REPORTS_LOG) as f:
        return {"reports": json.load(f)}


@app.post("/alert/trigger")
def trigger_alert(req: AlertRequest):
    msg = req.message or f"Landslide risk alert for zone {req.zone_id} (score {req.risk_score:.2f})"
    return _send_alert(req.zone_id, req.risk_score, msg)
