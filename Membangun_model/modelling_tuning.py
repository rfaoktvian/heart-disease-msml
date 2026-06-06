# modelling_tuning.py
# Kriteria 2 - Versi Lokal (tanpa DagsHub) untuk testing di lokal
# Autor: Nazly Rafa Oktafian Nuzqu

import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Heart-Disease-Local")


def load_and_preprocess(filepath='heart.csv'):
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
    X_test  = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def run_local(params, X_train, X_test, y_train, y_test, feature_names, run_name):
    with mlflow.start_run(run_name=run_name):
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy":           accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall_weighted":    recall_score(y_test, y_pred, average='weighted',    zero_division=0),
            "f1_weighted":        f1_score(y_test, y_pred, average='weighted',        zero_division=0),
            "f1_macro":           f1_score(y_test, y_pred, average='macro',           zero_division=0),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['No Disease', 'Disease'])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'CM - {run_name}')
        plt.tight_layout()
        cm_path = f'cm_{run_name}.png'
        plt.savefig(cm_path); plt.close()
        mlflow.log_artifact(cm_path, "plots")
        os.remove(cm_path)

        # Feature importance
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(feature_names)), importances[idx], color='steelblue')
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha='right')
        ax.set_title(f'Feature Importance - {run_name}')
        plt.tight_layout()
        fi_path = f'fi_{run_name}.png'
        plt.savefig(fi_path); plt.close()
        mlflow.log_artifact(fi_path, "plots")
        os.remove(fi_path)

        print(f"[{run_name}] Accuracy={metrics['accuracy']:.4f} | F1={metrics['f1_weighted']:.4f}")
        return metrics


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, feat = load_and_preprocess()
    configs = [
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 10,   "min_samples_split": 5},
        {"n_estimators": 300, "max_depth": 15,   "min_samples_split": 4},
    ]
    for i, p in enumerate(configs):
        run_local(p, X_train, X_test, y_train, y_test, feat,
                  run_name=f"local_run_{i+1}")
    print("\nSelesai! Buka MLflow UI: mlflow ui --port 5000")
