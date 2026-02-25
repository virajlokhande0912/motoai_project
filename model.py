"""
model.py — MOTOAI ML Model Trainer
Trains a RandomForestClassifier on cars.csv and saves it as model.pkl
Run this file once before starting app.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "cars.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def train_and_save():
    # ── 1. Load data ────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Loaded {len(df)} rows from cars.csv")
    print(df.head())

    # ── 2. Encode categorical columns ───────────────────────────
    encoders = {}
    categorical_cols = ["body_type", "fuel_type", "budget"]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"[INFO] Encoded '{col}' — classes: {list(le.classes_)}")

    # ── 3. Feature matrix & target ──────────────────────────────
    feature_cols = [
        "body_type_enc",
        "fuel_type_enc",
        "budget_enc",
        "seating",
        "safety",
        "comfort",
        "performance",
        "value",
    ]
    X = df[feature_cols]
    y = df["recommended_car"]

    # Encode target labels
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y)
    encoders["recommended_car"] = target_le

    # ── 4. Train / test split ───────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42
    )

    # ── 5. Train RandomForestClassifier ─────────────────────────
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    
    # ── 6. Evaluate ─────────────────────────────────────────────
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Accuracy: {acc * 100:.1f}%")
    print("\n[CLASSIFICATION REPORT]")

    # Fix for class mismatch issue
    labels = np.unique(np.concatenate((y_test, y_pred)))

    print(
        classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_le.inverse_transform(labels),
        zero_division=0,
    )
)

    # Feature importances
    importances = clf.feature_importances_
    print("\n[FEATURE IMPORTANCES]")
    for f, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        print(f"  {f:25s}: {imp:.4f}")

    # ── 7. Save model + encoders ─────────────────────────────────
    bundle = {"model": clf, "encoders": encoders, "feature_cols": feature_cols}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n[SAVED] model.pkl → {MODEL_PATH}")
    return acc


if __name__ == "__main__":
    train_and_save()
