"""
Synthetic training dataset generator for landslide risk in NER.

WHY SYNTHETIC: for the internal hackathon MVP you won't have time to clean
real multi-source data. This generates data with the SAME FEATURE STRUCTURE
real sources give you, so the exact same training script works once you swap
this file's output for real data.

REPLACE WITH REAL DATA LATER:
  - rainfall_mm_24h, rainfall_mm_7d   -> IMD gridded rainfall (imd.gov.in)
  - slope_deg, elevation_m            -> SRTM DEM (30m), processed with GDAL/rasterio
  - soil_moisture                     -> NASA SMAP or Bhuvan soil moisture product
  - historical_landslide_density      -> GSI Landslide Atlas + NASA COOLR (count per grid cell)
  - distance_to_road_km               -> OpenStreetMap road network
  - label (landslide_occurred)        -> GSI/COOLR historical event records

Run: python3 generate_dataset.py
Output: ../data/ner_landslide_training.csv
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 4000

NER_DISTRICTS = [
    "East Khasi Hills", "West Garo Hills", "Kohima", "Dima Hasao",
    "Upper Subansiri", "West Kameng", "Aizawl", "Churachandpur",
    "Gangtok", "Darjeeling-adjacent-NER",
]

df = pd.DataFrame({
    "district": np.random.choice(NER_DISTRICTS, N),
    "rainfall_mm_24h": np.random.gamma(2.0, 25, N).clip(0, 400),
    "rainfall_mm_7d": np.random.gamma(3.0, 60, N).clip(0, 1500),
    "slope_deg": np.random.beta(2.5, 2.0, N) * 60,          # 0-60 degrees
    "elevation_m": np.random.normal(1200, 600, N).clip(50, 3500),
    "soil_moisture": np.random.beta(2, 2, N),                 # 0-1 fraction
    "historical_landslide_density": np.random.exponential(0.4, N).clip(0, 5),
    "distance_to_road_km": np.random.exponential(2.5, N).clip(0, 20),
    "ndvi_vegetation_index": np.random.beta(3, 2, N),          # 0-1, higher = more vegetation cover
})

# Latent risk score built from realistic domain relationships:
# steeper slope + more rain + wetter soil + past landslide history + less vegetation = higher risk
risk_score = (
    0.030 * df.rainfall_mm_24h +
    0.008 * df.rainfall_mm_7d +
    0.045 * df.slope_deg +
    3.5   * df.soil_moisture +
    2.8   * df.historical_landslide_density +
    -2.0  * df.ndvi_vegetation_index +
    -0.15 * df.distance_to_road_km +
    np.random.normal(0, 1.5, N)   # noise
)

# Convert to binary label using a threshold tuned for a realistic ~18% positive rate
threshold = np.percentile(risk_score, 82)
df["landslide_occurred"] = (risk_score > threshold).astype(int)

df.to_csv("../data/ner_landslide_training.csv", index=False)
print(f"Generated {N} rows -> ../data/ner_landslide_training.csv")
print(f"Positive rate: {df['landslide_occurred'].mean():.1%}")
print(df.head())
