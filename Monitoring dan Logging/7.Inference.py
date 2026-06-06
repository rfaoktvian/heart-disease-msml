# 7.Inference.py
# Script uji inferensi ke API Heart Disease
# Autor: Nazly Rafa Oktafian Nuzqu

import requests
import time

API_URL = "http://localhost:8000/predict"

test_samples = [
    {
        "name": "Pasien 1 - Berisiko Tinggi",
        "data": {
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
        }
    },
    {
        "name": "Pasien 2 - Berisiko Rendah",
        "data": {
            "age": 37, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250,
            "fbs": 0, "restecg": 1, "thalach": 187, "exang": 0,
            "oldpeak": 3.5, "slope": 0, "ca": 0, "thal": 2
        }
    },
    {
        "name": "Pasien 3 - Perempuan Usia Pertengahan",
        "data": {
            "age": 56, "sex": 0, "cp": 1, "trestbps": 140, "chol": 294,
            "fbs": 0, "restecg": 0, "thalach": 153, "exang": 0,
            "oldpeak": 1.3, "slope": 1, "ca": 0, "thal": 2
        }
    },
]

def run_inference():
    print("=" * 55)
    print("  Heart Disease Inference Test")
    print("  Autor: Nazly Rafa Oktafian Nuzqu")
    print("=" * 55)

    try:
        health = requests.get("http://localhost:8000/health", timeout=5)
        print(f"[Health Check] {health.json()}\n")
    except Exception as e:
        print(f"[ERROR] API tidak dapat diakses: {e}")
        print("        Pastikan 3.prometheus_exporter.py sudah berjalan!")
        return

    for sample in test_samples:
        print(f"[TEST] {sample['name']}")
        try:
            start = time.time()
            resp = requests.post(API_URL, json=sample['data'], timeout=10)
            elapsed = time.time() - start

            if resp.status_code == 200:
                r = resp.json()
                print(f"  → Diagnosis     : {r['diagnosis']} (kelas {r['prediction']})")
                print(f"  → Probabilitas  : No Disease={r['probabilities']['No Disease']:.3f} | Disease={r['probabilities']['Disease']:.3f}")
                print(f"  → Latency       : {elapsed:.4f} detik")
            else:
                print(f"  → Error {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"  → Exception: {e}")
        print()

    print("[Prometheus Metrics Preview]")
    try:
        m = requests.get("http://localhost:8000/metrics", timeout=5)
        for line in m.text.split('\n'):
            if line.startswith('heart_') and not line.startswith('#'):
                print(f"  {line}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[INFO] Inference selesai!")

if __name__ == '__main__':
    run_inference()
