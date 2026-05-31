"""
Session-based evaluation utilities for motor predictive maintenance.
Matches the train/validation/test protocol in ieee-research-paper.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

# Sorted session folders map to session IDs 1-8 (paper numbering).
SESSION_ORDER = [
    "20240527_094865",
    "20240527_100759",
    "20240527_101627",
    "20240527_102436",
    "20240527_102919",
    "20240527_103311",
    "20240527_103690",
    "20240527_104247",
]

TRAIN_SESSION_IDS = {1, 2, 3, 5, 6}
VAL_SESSION_ID = 4
TEST_SESSION_IDS = {7, 8}

ROLLING_WINDOW = 5
RANDOM_STATE = 42


def session_name_to_id(session_name: str) -> int:
    return SESSION_ORDER.index(session_name) + 1


def session_id_to_name(session_id: int) -> str:
    return SESSION_ORDER[session_id - 1]


def get_session_split(session_name: str) -> str:
    session_id = session_name_to_id(session_name)
    if session_id in TRAIN_SESSION_IDS:
        return "train"
    if session_id == VAL_SESSION_ID:
        return "val"
    if session_id in TEST_SESSION_IDS:
        return "test"
    raise ValueError(f"Unknown session: {session_name}")


def engineer_features(
    df: pd.DataFrame,
    rolling_window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Build fused feature vector aligned with the paper (§III-E)."""
    out = df.copy()
    out["temp_rolling_mean"] = out.groupby(["session", "motor_id"])["temperature"].transform(
        lambda x: x.rolling(window=rolling_window, min_periods=1).mean()
    )
    out["voltage_rolling_std"] = out.groupby(["session", "motor_id"])["voltage"].transform(
        lambda x: x.rolling(window=rolling_window, min_periods=1).std()
    ).fillna(0)

    out["session_encoded"] = out["session"].map(lambda s: session_name_to_id(s) - 1)
    out["motor_encoded"] = out["motor_id"].str.extract(r"(\d+)").astype(int)[0] - 1
    return out


def feature_sets() -> Dict[str, List[str]]:
    base = ["temperature", "voltage", "position", "relative_time"]
    rolling = ["temp_rolling_mean", "voltage_rolling_std"]
    context = ["session_encoded", "motor_encoded"]
    return {
        "full": base + rolling + context,
        "raw_only": base + context,
        "raw_plus_temp_roll": base + ["temp_rolling_mean"] + context,
        "temperature_only": ["temperature", "relative_time"] + context,
        "voltage_only": ["voltage", "relative_time"] + context,
        "position_only": ["position", "relative_time"] + context,
    }


def split_by_session(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["split"] = df["session"].map(get_session_split)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    return train, val, test


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        score = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    return best_t


def train_random_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict:
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["is_anomaly"].astype(int).values
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df["is_anomaly"].astype(int).values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["is_anomaly"].astype(int).values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    t0 = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)
    train_time = time.perf_counter() - t0

    val_prob = model.predict_proba(X_val_s)[:, 1]
    threshold = best_f1_threshold(y_val, val_prob)
    test_prob = model.predict_proba(X_test_s)[:, 1]
    metrics = compute_metrics(y_test, test_prob, threshold)
    metrics["train_time_s"] = train_time

    y_pred = (test_prob >= threshold).astype(int)
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["feature_importance"] = dict(
        zip(feature_cols, model.feature_importances_.tolist())
    )
    metrics["_y_test"] = y_test
    metrics["_y_prob"] = test_prob
    return metrics


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict:
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["is_anomaly"].astype(int).values
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df["is_anomaly"].astype(int).values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["is_anomaly"].astype(int).values

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    t0 = time.perf_counter()
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = best_f1_threshold(y_val, val_prob)
    test_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_prob, threshold)
    metrics["train_time_s"] = train_time
    metrics["_y_test"] = y_test
    metrics["_y_prob"] = test_prob
    return metrics


def iqr_baseline_predict(df: pd.DataFrame) -> np.ndarray:
    """Apply global IQR OR-rule as a direct predictor (always predicts anomaly if rule fires)."""
    return df["is_anomaly"].astype(int).values


def isolation_forest_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    X_train = train_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    contamination = min(max(train_df["is_anomaly"].mean(), 0.01), 0.49)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_s)
    raw = model.predict(X_test_s)
    return (raw == -1).astype(int)


def prepare_lstm_sequences(
    df: pd.DataFrame,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sequences and return aligned session labels for split-aware partitioning."""
    sorted_data = df.sort_values(["session", "motor_id", "relative_time"]).reset_index(drop=True)
    sensor_features = ["temperature", "voltage", "position"]

    sequences: List[np.ndarray] = []
    labels: List[int] = []
    session_ids: List[int] = []

    for session in sorted_data["session"].unique():
        for motor in sorted_data["motor_id"].unique():
            motor_data = sorted_data[
                (sorted_data["session"] == session) & (sorted_data["motor_id"] == motor)
            ]
            if len(motor_data) < sequence_length:
                continue
            values = motor_data[sensor_features + ["is_anomaly"]].values
            sid = session_name_to_id(session)
            for i in range(len(values) - sequence_length + 1):
                sequences.append(values[i : i + sequence_length, :-1])
                labels.append(int(values[i + sequence_length - 1, -1]))
                session_ids.append(sid)

    return (
        np.array(sequences, dtype=np.float32),
        np.array(labels, dtype=int),
        np.array(session_ids, dtype=int),
    )


def train_lstm_session_split(
    df: pd.DataFrame,
    sequence_length: int = 30,
    epochs: int = 40,
) -> Dict:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import legacy as legacy_optimizers

    X, y, session_ids = prepare_lstm_sequences(df, sequence_length)
    train_mask = np.isin(session_ids, list(TRAIN_SESSION_IDS))
    val_mask = session_ids == VAL_SESSION_ID
    test_mask = np.isin(session_ids, list(TEST_SESSION_IDS))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        return {"error": "insufficient sequences", "sequence_length": sequence_length}

    scaler = StandardScaler()
    n_features = X_train.shape[-1]
    X_train_s = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_s = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    tf.random.set_seed(RANDOM_STATE)
    model = Sequential(
        [
            LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=legacy_optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy")

    t0 = time.perf_counter()
    model.fit(
        X_train_s,
        y_train,
        validation_data=(X_val_s, y_val),
        epochs=epochs,
        batch_size=64,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
    )
    train_time = time.perf_counter() - t0

    val_prob = model.predict(X_val_s, verbose=0).flatten()
    threshold = best_f1_threshold(y_val, val_prob)
    test_prob = model.predict(X_test_s, verbose=0).flatten()
    metrics = compute_metrics(y_test, test_prob, threshold)
    metrics["train_time_s"] = train_time
    metrics["sequence_length"] = sequence_length
    metrics["n_test_sequences"] = int(len(y_test))
    return metrics


def sensor_statistics(df: pd.DataFrame) -> Dict:
    stats = {}
    for sensor in ["temperature", "voltage", "position"]:
        col = df[sensor]
        outlier = (
            (col < col.quantile(0.25) - 1.5 * (col.quantile(0.75) - col.quantile(0.25)))
            | (col > col.quantile(0.75) + 1.5 * (col.quantile(0.75) - col.quantile(0.25)))
        )
        stats[sensor] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "outlier_rate_pct": float(outlier.mean() * 100),
        }
    return stats


def pca_motor_decomposition(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict:
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].fillna(0).values)
    X_test = scaler.transform(test_df[feature_cols].fillna(0).values)
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    pca.fit(X_train)
    test_pca = pca.transform(X_test)

    anomaly_mask = test_df["is_anomaly"].astype(bool).values
    anomaly_centroid = test_pca[anomaly_mask].mean(axis=0) if anomaly_mask.any() else np.zeros(3)

    per_motor = {}
    for motor in sorted(test_df["motor_id"].unique()):
        motor_mask = (test_df["motor_id"] == motor).values
        motor_anom = motor_mask & anomaly_mask
        dists = np.linalg.norm(test_pca[motor_anom] - anomaly_centroid, axis=1) if motor_anom.any() else []
        per_motor[motor] = {
            "test_anomaly_count": int(motor_anom.sum()),
            "test_anomaly_fraction_of_all_test_anomalies": float(
                motor_anom.sum() / max(anomaly_mask.sum(), 1)
            ),
            "median_pc_distance": float(np.median(dists)) if len(dists) else 0.0,
        }

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "per_motor": per_motor,
    }


def prepare_dataset(data_path: str = "data/raw/testing_data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from data_loader import MotorDataLoader

    loader = MotorDataLoader(data_path)
    loader.combine_all_data()
    labeled = loader.detect_anomalies(method="iqr")
    featured = engineer_features(labeled)
    train_df, val_df, test_df = split_by_session(featured)
    return train_df, val_df, test_df, labeled


def save_results(results: Dict, path: str = "results/session_eval_results.json") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    return out
