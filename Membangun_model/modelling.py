# modelling.py
# Kriteria 2 - Level Advance
# MLflow + DagsHub + Manual Logging + Hyperparameter Tuning
# Dataset: Heart Disease
# Autor: Nazly Rafa Oktafian Nuzqu

import json
import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# KONFIGURASI DAGSHUB
# ─────────────────────────────────────────────
DAGSHUB_USERNAME = os.environ.get("DAGSHUB_USERNAME", "rfaoktvian")   # ← sesuaikan
DAGSHUB_REPO     = os.environ.get("DAGSHUB_REPO",     "heart-disease-msml")  # ← sesuaikan
DAGSHUB_TOKEN    = os.environ.get("DAGSHUB_TOKEN",    "")

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
)
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

mlflow.set_experiment("Heart-Disease-Classification")


# ─────────────────────────────────────────────
# LOAD & PREPROCESSING
# ─────────────────────────────────────────────

def load_and_preprocess(filepath: str = 'heart.csv'):
    df = pd.read_csv(filepath).drop_duplicates().reset_index(drop=True)
    X = df.drop(columns=['target'])
    y = df['target']

    iso = IsolationForest(contamination=0.05, random_state=42)
    mask = iso.fit_predict(X) == 1
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()


# ─────────────────────────────────────────────
# ARTEFAK TAMBAHAN
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, run_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Disease', 'Disease'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion Matrix - {run_name}')
    plt.tight_layout()
    path = f'confusion_matrix_{run_name}.png'
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def plot_feature_importance(model, feature_names, run_name):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(feature_names)), importances[idx], color='steelblue')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha='right')
    ax.set_title(f'Feature Importance - {run_name}')
    ax.set_ylabel('Importance Score')
    plt.tight_layout()
    path = f'feature_importance_{run_name}.png'
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def save_classification_report(y_true, y_pred, run_name):
    report = classification_report(y_true, y_pred,
                                   target_names=['No Disease', 'Disease'])
    path = f'classification_report_{run_name}.txt'
    with open(path, 'w') as f:
        f.write(f"Classification Report - {run_name}\n{'='*50}\n{report}")
    return path


# ─────────────────────────────────────────────
# TRAINING + MANUAL LOGGING
# ─────────────────────────────────────────────

def train_and_log(params, X_train, X_test, y_train, y_test,
                  feature_names, run_name):
    with mlflow.start_run(run_name=run_name):
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy":           accuracy_score(y_test, y_pred),
            "precision_macro":    precision_score(y_test, y_pred, average='macro',    zero_division=0),
            "recall_macro":       recall_score(y_test, y_pred, average='macro',       zero_division=0),
            "f1_macro":           f1_score(y_test, y_pred, average='macro',           zero_division=0),
            "precision_weighted": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall_weighted":    recall_score(y_test, y_pred, average='weighted',    zero_division=0),
            "f1_weighted":        f1_score(y_test, y_pred, average='weighted',        zero_division=0),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="random_forest_model")

        # Artefak 1: Confusion Matrix
        cm_path = plot_confusion_matrix(y_test, y_pred, run_name)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Artefak 2: Feature Importance
        fi_path = plot_feature_importance(model, feature_names, run_name)
        mlflow.log_artifact(fi_path, artifact_path="plots")

        # Artefak 3: Classification Report
        cr_path = save_classification_report(y_test, y_pred, run_name)
        mlflow.log_artifact(cr_path, artifact_path="reports")

        # Artefak 4: Feature names JSON
        feat_json = f'feature_names_{run_name}.json'
        with open(feat_json, 'w') as f:
            json.dump({"feature_names": feature_names}, f, indent=2)
        mlflow.log_artifact(feat_json, artifact_path="metadata")

        # Cleanup
        for f in [cm_path, fi_path, cr_path, feat_json]:
            if os.path.exists(f):
                os.remove(f)

        print(f"[RUN] {run_name} | Accuracy={metrics['accuracy']:.4f} | F1={metrics['f1_weighted']:.4f}")
        return metrics


# ─────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

def hyperparameter_tuning(X_train, X_test, y_train, y_test, feature_names):
    param_grid = [
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 10,   "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 15,   "min_samples_split": 5},
        {"n_estimators": 300, "max_depth": 20,   "min_samples_split": 2},
        {"n_estimators": 300, "max_depth": None, "min_samples_split": 4},
    ]

    results = []
    for i, params in enumerate(param_grid):
        run_name = f"RF_run_{i+1}_ne{params['n_estimators']}_md{params['max_depth']}"
        metrics = train_and_log(params, X_train, X_test, y_train, y_test,
                                feature_names, run_name)
        results.append({"params": params, "metrics": metrics, "run_name": run_name})

    print("\n" + "=" * 60)
    print("RINGKASAN HYPERPARAMETER TUNING")
    print("=" * 60)
    best = max(results, key=lambda r: r['metrics']['f1_weighted'])
    for r in results:
        flag = " ← BEST" if r == best else ""
        print(f"{r['run_name']}: Acc={r['metrics']['accuracy']:.4f} | F1={r['metrics']['f1_weighted']:.4f}{flag}")
    print(f"\nParameter terbaik: {best['params']}")
    return best


if __name__ == '__main__':
    print("=" * 60)
    print("  KRITERIA 2 - ADVANCE | MLflow + DagsHub + Manual Logging")
    print("  Autor: Nazly Rafa Oktafian Nuzqu")
    print("=" * 60)

    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess('heart.csv')
    best = hyperparameter_tuning(X_train, X_test, y_train, y_test, feature_names)

    print("\n[INFO] Semua run berhasil dicatat ke DagsHub MLflow.")
