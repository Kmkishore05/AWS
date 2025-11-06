# train_model.py
# ------------------------------------------------------------
# Train an AI model to predict:
#   - Traffic congestion level (classification), or
#   - AQI / PM2.5 (regression)
#
# If 'sensor_data.csv' exists, it is used.
# Otherwise, a synthetic dataset is generated.
# ------------------------------------------------------------

import os
import math
import joblib
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer

# -------------------- CONFIG --------------------
# MODE = "traffic" for congestion classification (Low/Med/High)
# MODE = "aqi"     for AQI regression (predict PM2.5 as proxy)
MODE = "traffic"        # change to "aqi" for regression
DATA_PATH = "sensor_data.csv"  # optional existing data file
TARGET_COL_TRAFFIC = "congestion"  # categorical target
TARGET_COL_AQI = "pm25"            # numeric target
RANDOM_SEED = 42
N_SAMPLES_SYNTH = 12000            # synthetic fallback
MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(exist_ok=True)
# ------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)


def generate_synthetic_data(n=N_SAMPLES_SYNTH) -> pd.DataFrame:
    """Generate a realistic synthetic dataset combining traffic + air-quality signals."""
    base_time = datetime(2025, 1, 1, 0, 0, 0)
    rows = []
    locations = ["Main Road", "2nd Avenue", "Central Park", "Tech Park", "Airport Rd"]

    for i in range(n):
        ts = base_time + timedelta(minutes=int(i % (24 * 60)))
        hour = ts.hour
        dow = ts.weekday()  # 0=Mon

        location = rng.choice(locations)
        is_holiday = 1 if (dow >= 5 and rng.random() < 0.2) else 0  # some weekends are holiday-like

        # Traffic features
        rush_factor = 1.0
        if 7 <= hour <= 10 or 17 <= hour <= 20:
            rush_factor = 1.6
        if is_holiday:
            rush_factor *= 0.7

        # base flow by location
        base_flow = {
            "Main Road": 750,
            "2nd Avenue": 450,
            "Central Park": 200,
            "Tech Park": 600,
            "Airport Rd": 900,
        }[location]

        traffic_flow = max(
            50,
            rng.normal(base_flow * rush_factor, 80)
        )  # vehicles/hour

        avg_speed = max(
            5.0,
            rng.normal(45 - 0.02 * traffic_flow, 5.0)
        )  # km/h (decreases with flow)

        # Weather/air features
        temperature = rng.normal(30 - 0.1 * (month_from_ts := ts.month) + 0.5 * math.sin(hour / 24 * 2 * math.pi), 3)
        humidity = np.clip(rng.normal(55 + 10 * math.sin(hour / 24 * 2 * math.pi), 12), 20, 95)
        wind = np.clip(rng.normal(8, 3), 0.1, 25)  # km/h

        # Emissions proxies
        no2 = np.clip(rng.normal(25 + 0.03 * traffic_flow, 8), 3, 150)
        co = np.clip(rng.normal(0.3 + 0.0003 * traffic_flow, 0.1), 0.05, 5.0)
        pm10 = np.clip(rng.normal(40 + 0.02 * traffic_flow - 0.7 * wind, 10), 5, 300)
        pm25 = np.clip(0.55 * pm10 + rng.normal(5, 6), 3, 250)

        # Derive congestion label
        if avg_speed >= 35 and traffic_flow < 500:
            congestion = "Low"
        elif avg_speed >= 20 and traffic_flow < 800:
            congestion = "Medium"
        else:
            congestion = "High"

        rows.append(
            dict(
                timestamp=ts,
                location=location,
                traffic_flow=round(float(traffic_flow), 2),
                avg_speed=round(float(avg_speed), 2),
                temperature=round(float(temperature), 2),
                humidity=round(float(humidity), 2),
                wind=round(float(wind), 2),
                no2=round(float(no2), 2),
                co=round(float(co), 3),
                pm10=round(float(pm10), 2),
                pm25=round(float(pm25), 2),
                is_holiday=is_holiday,
                dayofweek=dow,
                hour=hour,
                congestion=congestion,
            )
        )

    df = pd.DataFrame(rows)
    return df


def load_data(path=DATA_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)
        # Fill in day/hour if missing
        if "dayofweek" not in df:
            df["dayofweek"] = pd.to_datetime(df["timestamp"]).dt.weekday
        if "hour" not in df:
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        return df
    else:
        print(f"[info] '{path}' not found. Generating synthetic dataset...")
        return generate_synthetic_data()


def build_pipeline(mode: str, categorical: list, numeric: list):
    # Preprocess: impute + one-hot for categoricals; impute for numerics
    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical),
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric),
        ]
    )
    if mode == "traffic":
        model = RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=RANDOM_SEED, n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=400, max_depth=None, random_state=RANDOM_SEED, n_jobs=-1
        )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    return pipe


def main():
    df = load_data()

    # Select features
    feature_cols = [
        "location", "traffic_flow", "avg_speed",
        "temperature", "humidity", "wind",
        "no2", "co", "pm10", "pm25",  # keep pm25 as a feature for traffic mode
        "is_holiday", "dayofweek", "hour",
    ]

    # For AQI regression, we don't want to leak pm25 into the target
    if MODE == "aqi":
        feature_cols = [c for c in feature_cols if c != "pm25"]

    categorical = ["location", "is_holiday", "dayofweek", "hour"]
    numeric = [c for c in feature_cols if c not in categorical]

    if MODE == "traffic":
        target = TARGET_COL_TRAFFIC
        df = df.dropna(subset=[target])
    else:
        target = TARGET_COL_AQI
        df = df.dropna(subset=[target])

    X = df[feature_cols].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y if MODE == "traffic" else None
    )

    pipe = build_pipeline(MODE, categorical, numeric)

    print(f"[info] Training '{MODE}' model...")
    pipe.fit(X_train, y_train)

    # Evaluation
    y_pred = pipe.predict(X_test)

    if MODE == "traffic":
        print("\n=== Classification Report (Traffic Congestion) ===")
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
        cv = cross_val_score(pipe, X, y, cv=5, scoring="f1_weighted", n_jobs=-1)
        print(f"5-fold CV F1 (weighted): {cv.mean():.3f} ± {cv.std():.3f}")
    else:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        print("\n=== Regression Metrics (AQI/PM2.5) ===")
        print(f"MAE : {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R^2 : {r2:.3f}")
        cv = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)
        print(f"5-fold CV R^2: {cv.mean():.3f} ± {cv.std():.3f}")

    # Save
    model_path = MODELS_DIR / f"{'congestion' if MODE=='traffic' else 'aqi'}_model.joblib"
    joblib.dump(pipe, model_path)
    print(f"[info] Saved model to: {model_path}")

    # Quick demo prediction
    demo = X_test.iloc[[0]].copy()
    pred = pipe.predict(demo)[0]
    print("\n[demo] Input sample:")
    print(demo.to_dict(orient="records")[0])
    print(f"[demo] Predicted {('congestion' if MODE=='traffic' else 'pm25')}: {pred}")

    # Feature importance (approx via permutation on RF inputs after OHE is tricky).
    # For a quick feel, we can show training feature importances using model attribute
    # by mapping back to preprocessed columns:
    try:
        rf = pipe.named_steps["model"]
        pre = pipe.named_steps["pre"]
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        cat_names = ohe.get_feature_names_out(categorical)
        feature_names = list(cat_names) + numeric
        importances = rf.feature_importances_
        topk = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:15]
        print("\n[top features]")
        for name, imp in topk:
            print(f"{name:30s} : {imp:.4f}")
    except Exception as e:
        print(f"[warn] Could not compute feature importances: {e}")


if __name__ == "__main__":
    # Dependencies:
    #   pip install numpy pandas scikit-learn joblib
    main()
