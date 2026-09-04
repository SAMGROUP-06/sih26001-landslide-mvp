"""
Trains a landslide risk classifier and saves it for the FastAPI backend to serve.

Run: python3 train_model.py
Output: ../models/landslide_risk_model.joblib
"""
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

FEATURES = [
    "rainfall_mm_24h", "rainfall_mm_7d", "slope_deg", "elevation_m",
    "soil_moisture", "historical_landslide_density", "distance_to_road_km",
    "ndvi_vegetation_index",
]
TARGET = "landslide_occurred"

df = pd.read_csv("../data/ner_landslide_training.csv")
X, y = df[FEATURES], df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=== Model Performance (on held-out test set) ===")
print(classification_report(y_test, y_pred, target_names=["Safe", "Landslide Risk"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

print("\n=== Feature Importance ===")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:32s} {imp:.3f}")

joblib.dump({"model": model, "features": FEATURES}, "../models/landslide_risk_model.joblib")
print("\nSaved -> ../models/landslide_risk_model.joblib")
