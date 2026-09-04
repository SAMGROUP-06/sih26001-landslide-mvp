import os
import json
import random
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="SIH26001 - Landslide Risk Monitoring API",
    description="Landslide Risk Monitoring MVP Backend",
    version="1.0.0",
)


# Allow Netlify frontend and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# FILE CONFIGURATION
# ============================================================

REPORTS_LOG = "citizen_reports.json"

RISK_ALERT_THRESHOLD = 0.70


# ============================================================
# DATA MODELS
# ============================================================

class PredictInput(BaseModel):
    rainfall: float = 50.0
    soil_moisture: float = 50.0
    slope: float = 30.0
    temperature: float = 25.0
    humidity: float = 70.0


class AlertRequest(BaseModel):
    zone_id: str
    risk_score: float
    message: Optional[str] = None


# ============================================================
# ZONE DATA
# ============================================================

ZONES = [
    {
        "id": "zone_01",
        "name": "Himalayan Zone A",
        "lat": 30.0668,
        "lon": 79.0193,
    },
    {
        "id": "zone_02",
        "name": "Himalayan Zone B",
        "lat": 30.3165,
        "lon": 78.0322,
    },
    {
        "id": "zone_03",
        "name": "Mountain Zone C",
        "lat": 31.1048,
        "lon": 77.1734,
    },
    {
        "id": "zone_04",
        "name": "Hill Zone D",
        "lat": 32.2432,
        "lon": 77.1892,
    },
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _risk_level(risk: float) -> str:
    """
    Convert risk score into a human-readable risk level.
    """

    if risk >= 0.70:
        return "HIGH"

    if risk >= 0.40:
        return "MEDIUM"

    return "LOW"


def _random_live_features() -> dict:
    """
    Generate demo sensor/weather values.
    """

    return {
        "rainfall": round(random.uniform(10, 180), 2),
        "soil_moisture": round(random.uniform(20, 95), 2),
        "slope": round(random.uniform(5, 60), 2),
        "temperature": round(random.uniform(5, 35), 2),
        "humidity": round(random.uniform(30, 100), 2),
    }


def _fallback_risk(features: dict) -> float:
    """
    Calculate a simple demo risk score.

    This is NOT a real ML model.
    It is only for MVP/demo purposes.
    """

    rainfall = float(features.get("rainfall", 50))
    soil_moisture = float(features.get("soil_moisture", 50))
    slope = float(features.get("slope", 30))
    humidity = float(features.get("humidity", 70))

    rainfall_score = min(rainfall / 150.0, 1.0)
    moisture_score = min(soil_moisture / 100.0, 1.0)
    slope_score = min(slope / 60.0, 1.0)
    humidity_score = min(humidity / 100.0, 1.0)

    risk = (
        rainfall_score * 0.40
        + moisture_score * 0.25
        + slope_score * 0.25
        + humidity_score * 0.10
    )

    return round(max(0.0, min(risk, 1.0)), 3)


def _predict(features: dict) -> float:
    """
    Prediction function.

    If you later add a trained ML model, this function
    can be replaced with actual model prediction.
    """

    return _fallback_risk(features)


def _log_json(filename: str, entry: dict):
    """
    Save a citizen report to a JSON file.
    """

    try:
        existing_data = []

        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    existing_data = data

            except (json.JSONDecodeError, OSError):
                existing_data = []

        existing_data.append(entry)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                existing_data,
                f,
                indent=2,
                ensure_ascii=False,
            )

    except Exception as e:
        print(f"[LOG ERROR] {type(e).__name__}: {e}")


def _send_alert(
    zone_id: str,
    risk_score: float,
    message: str,
) -> dict:
    """
    Demo alert function.

    Currently prints the alert to the Render logs.
    Later this can be connected to SMS/email/WhatsApp.
    """

    alert = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "zone_id": zone_id,
        "risk_score": round(risk_score, 3),
        "risk_level": _risk_level(risk_score),
        "message": message,
        "status": "alert_triggered",
    }

    print("=" * 60)
    print("LANDSLIDE ALERT")
    print("=" * 60)
    print(f"Zone: {zone_id}")
    print(f"Risk Score: {risk_score:.3f}")
    print(f"Risk Level: {_risk_level(risk_score)}")
    print(f"Message: {message}")
    print("=" * 60)

    return alert


# ============================================================
# ROOT / HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "project": "SIH26001 - Landslide Risk Monitoring MVP",
        "message": "Landslide Risk Monitoring API is running.",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "landslide-risk-api",
    }


# ============================================================
# ZONE RISK DASHBOARD
# ============================================================

@app.get("/zones")
def get_zones():

    results = []

    for z in ZONES:

        try:
            # Generate demo live sensor data
            features = _random_live_features()

            # Calculate risk
            risk = _predict(features)

            entry = {
                **z,
                "risk_score": risk,
                "features": features,
                "risk_level": _risk_level(risk),
            }

            # Trigger alert if risk is high
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
            print(
                f"[ZONE ERROR] "
                f"{z['name']}: "
                f"{type(e).__name__}: {e}"
            )

            fallback_features = _random_live_features()
            fallback_risk = _fallback_risk(fallback_features)

            results.append(
                {
                    **z,
                    "risk_score": fallback_risk,
                    "features": fallback_features,
                    "risk_level": _risk_level(fallback_risk),
                    "warning": "Demo fallback score used.",
                }
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "zones": results,
    }


# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.post("/predict")
def predict(inp: PredictInput):

    # Pydantic v2 compatible
    features = inp.model_dump()

    risk = _predict(features)

    return {
        "risk_score": round(risk, 3),
        "risk_level": _risk_level(risk),
        "features": features,
    }


# ============================================================
# CITIZEN REPORT
# ============================================================

@app.post("/citizen-report")
async def citizen_report(
    lat: float = Form(...),
    lon: float = Form(...),
    description: str = Form(...),
    photo: Optional[UploadFile] = File(None),
):

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lat": lat,
        "lon": lon,
        "description": description,
        "photo_filename": photo.filename if photo else None,
    }

    _log_json(REPORTS_LOG, entry)

    return {
        "status": "received",
        "message": "Citizen report received successfully.",
        "report": entry,
    }


# ============================================================
# GET CITIZEN REPORTS
# ============================================================

@app.get("/citizen-reports")
def list_reports():

    if not os.path.exists(REPORTS_LOG):
        return {
            "reports": []
        }

    try:

        with open(
            REPORTS_LOG,
            "r",
            encoding="utf-8",
        ) as f:

            reports = json.load(f)

        if not isinstance(reports, list):
            reports = []

        return {
            "reports": reports
        }

    except Exception as e:

        print(
            f"[REPORTS ERROR] "
            f"{type(e).__name__}: {e}"
        )

        return {
            "reports": []
        }


# ============================================================
# MANUAL ALERT TRIGGER
# ============================================================

@app.post("/alert/trigger")
def trigger_alert(req: AlertRequest):

    msg = req.message or (
        f"Landslide risk alert for zone "
        f"{req.zone_id} "
        f"(score {req.risk_score:.2f})"
    )

    return _send_alert(
        req.zone_id,
        req.risk_score,
        msg,
    )


# ============================================================
# DEMO STATUS
# ============================================================

@app.get("/demo")
def demo():

    features = _random_live_features()
    risk = _predict(features)

    return {
        "message": "Demo prediction generated.",
        "features": features,
        "risk_score": risk,
        "risk_level": _risk_level(risk),
    }
