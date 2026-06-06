# 3.prometheus_exporter.py
# Kriteria 4 - Level Basic
# Serving model Heart Disease + Prometheus metrics
# Autor: Nazly Rafa Oktafian Nuzqu
#
# Cara menjalankan:
#   python 3.prometheus_exporter.py
#
# Endpoint:
#   POST http://localhost:8000/predict
#   GET  http://localhost:8000/metrics

import time
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ─────────────────────────────────────────────
# PROMETHEUS METRICS
# ─────────────────────────────────────────────

PREDICTION_COUNTER = Counter(
    'heart_prediction_requests_total',
    'Total jumlah request prediksi',
    ['status']
)

PREDICTION_LATENCY = Histogram(
    'heart_prediction_latency_seconds',
    'Latensi prediksi dalam detik',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

PREDICTION_CLASS_COUNTER = Counter(
    'heart_prediction_class_total',
    'Jumlah prediksi per kelas',
    ['diagnosis']
)

ACTIVE_REQUESTS = Gauge(
    'heart_active_requests',
    'Jumlah request aktif saat ini'
)

MODEL_LOAD_STATUS = Gauge(
    'heart_model_loaded',
    '1 jika model berhasil dimuat'
)

# ─────────────────────────────────────────────
# FEATURE NAMES (13 fitur Heart Disease)
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal'
]

CLASS_NAMES = {0: 'No Disease', 1: 'Disease'}

# ─────────────────────────────────────────────
# LOAD / BUAT MODEL
# ─────────────────────────────────────────────

def build_model():
    """Buat dan latih model dari heart.csv jika ada, atau buat dummy."""
    if os.path.exists('heart.csv'):
        print("[INFO] Melatih model dari heart.csv...")
        df = pd.read_csv('heart.csv').drop_duplicates().reset_index(drop=True)
        X = df.drop(columns=['target'])
        y = df['target']
        iso  = IsolationForest(contamination=0.05, random_state=42)
        mask = iso.fit_predict(X) == 1
        X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        model.fit(X_train_scaled, y_train)
        print("[INFO] Model berhasil dilatih dari heart.csv")
    else:
        print("[WARN] heart.csv tidak ditemukan. Membuat dummy model...")
        from sklearn.datasets import make_classification
        X_dummy, y_dummy = make_classification(
            n_samples=300, n_features=13, n_classes=2, random_state=42
        )
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_dummy)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_scaled, y_dummy)
        print("[INFO] Dummy model berhasil dibuat.")
    return model, scaler

model, scaler = build_model()
MODEL_LOAD_STATUS.set(1 if model is not None else 0)

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint prediksi heart disease.
    Contoh request body:
    {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    """
    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        data = request.get_json(force=True)

        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            PREDICTION_COUNTER.labels(status='error').inc()
            return jsonify({"error": f"Missing features: {missing}"}), 400

        input_df     = pd.DataFrame([data])[FEATURE_NAMES]
        input_scaled = scaler.transform(input_df)

        prediction    = int(model.predict(input_scaled)[0])
        probabilities = model.predict_proba(input_scaled)[0].tolist()
        class_name    = CLASS_NAMES[prediction]

        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_COUNTER.labels(status='success').inc()
        PREDICTION_CLASS_COUNTER.labels(diagnosis=class_name).inc()

        return jsonify({
            "prediction":  prediction,
            "diagnosis":   class_name,
            "probabilities": {
                "No Disease": round(probabilities[0], 4),
                "Disease":    round(probabilities[1], 4),
            },
            "latency_seconds": round(latency, 4)
        })

    except Exception as e:
        PREDICTION_COUNTER.labels(status='error').inc()
        return jsonify({"error": str(e)}), 500

    finally:
        ACTIVE_REQUESTS.dec()


@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == '__main__':
    print("=" * 55)
    print("  Heart Disease Prediction API")
    print("  Autor: Nazly Rafa Oktafian Nuzqu")
    print("  Predict : POST http://localhost:8000/predict")
    print("  Metrics : GET  http://localhost:8000/metrics")
    print("  Health  : GET  http://localhost:8000/health")
    print("=" * 55)
    app.run(host='0.0.0.0', port=8000, debug=False)
