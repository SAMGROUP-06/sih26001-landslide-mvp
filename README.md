# SIH26001 — Landslide Risk Monitoring MVP (starter code)

This is a working starter pipeline: **data → ML model → API → alerts**.
It runs end-to-end right now on synthetic data, so you have a real demo
immediately — then you swap in real datasets as time allows.

## What's in here

```
ml/
  generate_dataset.py   -> makes synthetic training data (realistic feature structure)
  train_model.py         -> trains the risk model, prints accuracy, saves it
data/
  ner_landslide_training.csv   -> generated dataset (already created)
models/
  landslide_risk_model.joblib  -> trained model (already created)
backend/
  main.py                -> FastAPI server: predictions, alerts, citizen reports
  requirements.txt
```

The **GIS dashboard** was built separately as an interactive artifact in the
chat — screenshot/record that for your demo video, or rebuild it as a React
page using the same `/zones` API shape (see "Wiring the dashboard to the
real API" below).

## Quick start (on your own laptop, needs internet)

```bash
# 1. ML — already run once, re-run any time you change the dataset
cd ml
python3 generate_dataset.py
python3 train_model.py

# 2. Backend
cd ../backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open http://localhost:8000/zones in a browser — you'll get live JSON
risk scores for 8 demo NER districts, e.g.:

```json
{"zones": [{"id": "z1", "name": "East Khasi Hills", "risk_score": 0.71, "risk_level": "high", ...}]}
```

Any zone scoring >= 0.6 auto-triggers an alert (logged to
`backend/alert_log.json`, or sent as a real SMS if you set Twilio env vars).

## Swapping in real data (do this once the core pipeline is proven)

Replace `data/ner_landslide_training.csv` with real data. Column names must
match `FEATURES` in `train_model.py`:

| Column | Real source |
|---|---|
| `rainfall_mm_24h`, `rainfall_mm_7d` | IMD gridded rainfall — imd.gov.in |
| `slope_deg`, `elevation_m` | SRTM DEM (30m) — process with `rasterio`/GDAL |
| `soil_moisture` | NASA SMAP, or Bhuvan (ISRO) soil moisture product |
| `historical_landslide_density` | GSI Landslide Atlas + NASA COOLR catalog |
| `distance_to_road_km` | OpenStreetMap road network via `osmnx` |
| `ndvi_vegetation_index` | Sentinel-2 / MODIS NDVI (Google Earth Engine free tier) |
| `landslide_occurred` (label) | GSI/COOLR historical event records |

Then just re-run `train_model.py` — nothing else changes.

## Real SMS alerts (optional, takes 5 minutes)

Sign up for a free Twilio trial account, then:

```bash
export TWILIO_SID=xxxx
export TWILIO_TOKEN=xxxx
export TWILIO_FROM=+1xxxxxxxxxx      # Twilio number
export ALERT_TO=+91xxxxxxxxxx        # your phone, for the demo
```

Restart the backend — alerts now send real SMS instead of just logging.

## Wiring the dashboard to the real API

The chat artifact currently scores zones client-side (so it works standalone
with no backend). To make it pull real predictions, replace the
`scoreZone()` function's math with a `fetch("http://localhost:8000/zones")`
call and map the response into the same `data` array shape used by the
markers. Point `report-submit` at `POST /citizen-report` (multipart form:
`lat`, `lon`, `description`, optional `photo`) instead of just logging
locally.

## For the judges — talking points

- Model trained on realistic feature relationships; swap-in point for real
  data is a single CSV, no code changes needed elsewhere.
- Full loop works today: data → prediction → map → alert — not just slides.
- Alerting is real (Twilio), not simulated, once env vars are set.
- Everything else (satellite integration, government data-sharing MOUs) is
  explicitly future scope — said upfront in the feasibility slide, not
  oversold.
