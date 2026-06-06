# train.py
# Entry point untuk MLflow Project (Kriteria 3)
# Dipanggil otomatis oleh GitHub Actions CI
# Autor: Nazly Rafa Oktafian Nuzqu

import argparse
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

# ── Argparse ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators",      type=int,   default=200)
parser.add_argument("--max_depth",         type=int,   default=15)
parser.add_argument("--min_samples_split", type=int,   default=5)
parser.add_argument("--test_size",         type=float, default=0.2)
parser.add_argument("--contamination",     type=float, default=0.05)
args = parser.parse_args()

# ── MLflow + DagsHub Setup (tanpa dagshub.init) ───────────────────────────────
DAGSHUB_USERNAME = os.environ.get("DAGSHUB_USERNAME", "rfaoktvian")
DAGSHUB_REPO     = os.environ.get("DAGSHUB_REPO",     "heart-disease-msml")
DAGSHUB_TOKEN    = os.environ.get("DAGSHUB_TOKEN",    "")

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
)
mlflow.set_experiment("Heart-Disease-CI")

# ── Data Loading & Preprocessing ──────────────────────────────────────────────
df = pd.read_csv("heart.csv").drop_duplicates().reset_index(drop=True)
X = df.drop(columns=["target"])
y = df["target"]

iso  = IsolationForest(contamination=args.contamination, random_state=42)
mask = iso.fit_predict(X) == 1
X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=42, stratify=y
)
scaler  = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

# ── Training & Manual Logging ──────────────────────────────────────────────────


params = {
    "n_estimators":      args.n_estimators,
    "max_depth":         args.max_depth,
    "min_samples_split": args.min_samples_split,
}

model  = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

metrics = {
    "accuracy":           accuracy_score(y_test, y_pred),
    "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
    "recall_weighted":    recall_score(y_test, y_pred, average="weighted",    zero_division=0),
    "f1_weighted":        f1_score(y_test, y_pred, average="weighted",        zero_division=0),
    "f1_macro":           f1_score(y_test, y_pred, average="macro",           zero_division=0),
    "precision_macro":    precision_score(y_test, y_pred, average="macro",    zero_division=0),
    "recall_macro":       recall_score(y_test, y_pred, average="macro",       zero_division=0),
}

mlflow.log_params(params)
mlflow.log_metrics(metrics)
mlflow.sklearn.log_model(model, artifact_path="model")

# Confusion matrix
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix - CI Run")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
plt.close()
mlflow.log_artifact("confusion_matrix.png", "plots")

# Feature importance
feat_names  = X.columns.tolist()
importances = model.feature_importances_
idx = np.argsort(importances)[::-1]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(feat_names)), importances[idx], color="steelblue")
ax.set_xticks(range(len(feat_names)))
ax.set_xticklabels([feat_names[i] for i in idx], rotation=45, ha="right")
ax.set_title("Feature Importance - CI Run")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=100)
plt.close()
mlflow.log_artifact("feature_importance.png", "plots")

# Classification report
report = classification_report(y_test, y_pred,
                                target_names=["No Disease", "Disease"])
with open("classification_report.txt", "w") as f:
    f.write(report)
mlflow.log_artifact("classification_report.txt", "reports")

# Feature names JSON
with open("feature_names.json", "w") as f:
    json.dump({"feature_names": feat_names}, f, indent=2)
mlflow.log_artifact("feature_names.json", "metadata")


print(f"[CI] Accuracy   : {metrics['accuracy']:.4f}")
print(f"[CI] F1 Weighted: {metrics['f1_weighted']:.4f}")
print("[CI] Semua artefak berhasil di-log ke DagsHub MLflow.")
