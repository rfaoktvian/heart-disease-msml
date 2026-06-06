# automate_Nazly-Rafa-Oktafian-Nuzqu.py
# Skrip otomatisasi preprocessing untuk Heart Disease Dataset
# Kriteria 1 - Level Skilled
# Autor: Nazly Rafa Oktafian Nuzqu

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import os
import argparse


def load_data(filepath: str) -> pd.DataFrame:
    print(f"[INFO] Memuat dataset dari: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Shape dataset: {df.shape}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"[INFO] Duplikat dihapus: {before - after} baris | Sisa: {after} baris")
    return df


def remove_outliers(X: pd.DataFrame, y: pd.Series,
                    contamination: float = 0.05) -> tuple:
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(X)
    mask = labels == 1
    X_clean = X[mask].reset_index(drop=True)
    y_clean = y[mask].reset_index(drop=True)
    print(f"[INFO] Outlier dihapus: {(~mask).sum()} baris | Sisa: {len(X_clean)} baris")
    return X_clean, y_clean


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> tuple:
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )
    print("[INFO] Standarisasi fitur selesai.")
    return X_train_scaled, X_test_scaled, scaler


def save_preprocessed(X_train, X_test, y_train, y_test,
                       output_dir: str = '.') -> None:
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'heart_train_preprocessed.csv')
    test_path  = os.path.join(output_dir, 'heart_test_preprocessed.csv')

    train_df = X_train.copy()
    train_df['target'] = y_train.values
    train_df.to_csv(train_path, index=False)

    test_df = X_test.copy()
    test_df['target'] = y_test.values
    test_df.to_csv(test_path, index=False)

    print(f"[INFO] Dataset preprocessed disimpan:")
    print(f"       - {train_path}")
    print(f"       - {test_path}")


def run_preprocessing_pipeline(input_file: str,
                                output_dir: str = 'preprocessed_data',
                                contamination: float = 0.05,
                                test_size: float = 0.2) -> None:
    print("=" * 55)
    print("   PIPELINE PREPROCESSING - Heart Disease Dataset")
    print("   Autor: Nazly Rafa Oktafian Nuzqu")
    print("=" * 55)

    df = load_data(input_file)
    df = remove_duplicates(df)

    X = df.drop(columns=['target'])
    y = df['target']

    X, y = remove_outliers(X, y, contamination=contamination)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    save_preprocessed(X_train_scaled, X_test_scaled, y_train, y_test,
                      output_dir=output_dir)

    print("\n[INFO] Pipeline preprocessing selesai!")
    print("=" * 55)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Automate preprocessing untuk Heart Disease Dataset'
    )
    parser.add_argument('--input',         type=str,   default='heart.csv',
                        help='Path ke file CSV input (default: heart.csv)')
    parser.add_argument('--output',        type=str,   default='preprocessed_data',
                        help='Folder output (default: preprocessed_data)')
    parser.add_argument('--contamination', type=float, default=0.05)
    parser.add_argument('--test_size',     type=float, default=0.2)
    args = parser.parse_args()

    run_preprocessing_pipeline(
        input_file=args.input,
        output_dir=args.output,
        contamination=args.contamination,
        test_size=args.test_size
    )
