#!/usr/bin/env python3
"""Fast paper evaluation: RF/XGB/ablations/figures without LSTM training."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from session_eval import (  # noqa: E402
    SESSION_ORDER,
    TEST_SESSION_IDS,
    TRAIN_SESSION_IDS,
    VAL_SESSION_ID,
    compute_metrics,
    feature_sets,
    iqr_baseline_predict,
    isolation_forest_predict,
    pca_motor_decomposition,
    prepare_dataset,
    sensor_statistics,
    train_random_forest,
    train_xgboost,
)

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results" / "session_eval_results.json"


def run_ablations(train_df, val_df, test_df, full_features):
    rows = []
    y_test = test_df["is_anomaly"].astype(int).values
    y_iqr = iqr_baseline_predict(test_df)
    rows.append({
        "experiment": "IQR OR-rule (label rule as predictor)",
        "roc_auc": None, "pr_auc": None,
        "precision": round(float((y_iqr & y_test).sum() / max(y_iqr.sum(), 1)), 3),
        "recall": round(float((y_iqr & y_test).sum() / max(y_test.sum(), 1)), 3),
        "f1": round(float(2 * (y_iqr & y_test).sum() / max(y_iqr.sum() + y_test.sum(), 1)), 3),
        "train_time_s": 0.0,
    })
    iso = isolation_forest_predict(train_df, test_df, full_features)
    im = compute_metrics(y_test, iso.astype(float), 0.5)
    rows.append({"experiment": "Isolation Forest", **{k: round(im[k], 3) for k in ["roc_auc", "pr_auc", "precision", "recall", "f1"]}, "train_time_s": 0.0})
    for name, cols in {
        "RF — temperature only": feature_sets()["temperature_only"],
        "RF — voltage only": feature_sets()["voltage_only"],
        "RF — position only": feature_sets()["position_only"],
        "RF — raw channels only": feature_sets()["raw_only"],
        "RF — raw + rolling mean": feature_sets()["raw_plus_temp_roll"],
        "RF — full feature set": feature_sets()["full"],
    }.items():
        m = train_random_forest(train_df, val_df, test_df, cols)
        rows.append({"experiment": name, **{k: round(m[k], 3) for k in ["roc_auc", "pr_auc", "precision", "recall", "f1", "train_time_s"]}})
    return rows


def generate_figures(train_df, test_df, full_features, rf_metrics, xgb_metrics):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 14})

    imp = pd.Series(rf_metrics["feature_importance"]).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    imp.plot(kind="barh", ax=ax, color="#2E86AB")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance (Session Split)")
    fig.savefig(FIGURES / "ieee_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(train_df[full_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.savefig(FIGURES / "ieee_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_df[full_features].fillna(0).values)
    X_test_s = scaler.transform(test_df[full_features].fillna(0).values)
    pca = PCA(n_components=3, random_state=42)
    pca.fit(X_train_s)
    test_pca = pca.transform(X_test_s)
    y_test = test_df["is_anomaly"].astype(int).values
    motors = test_df["motor_id"].values

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(test_pca[y_test == 0, 0], test_pca[y_test == 0, 1], test_pca[y_test == 0, 2], c="#A8DADC", s=8, alpha=0.5, label="Normal")
    ax1.scatter(test_pca[y_test == 1, 0], test_pca[y_test == 1, 1], test_pca[y_test == 1, 2], c="#E63946", s=12, alpha=0.7, label="Proxy anomaly")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax1.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    ax1.set_title("PCA by Anomaly Label")
    ax1.legend()
    ax2 = fig.add_subplot(122, projection="3d")
    for motor in sorted(set(motors)):
        m = motors == motor
        ax2.scatter(test_pca[m, 0], test_pca[m, 1], test_pca[m, 2], s=10, alpha=0.6, label=motor.replace("data_motor_", "M"))
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax2.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    ax2.set_title("PCA by Motor ID (Test Set)")
    ax2.legend(fontsize=8)
    fig.savefig(FIGURES / "ieee_3d_pca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for label, metrics, color in [("Random Forest", rf_metrics, "#2E86AB"), ("XGBoost", xgb_metrics, "#F18F01")]:
        fpr, tpr, _ = roc_curve(metrics["_y_test"], metrics["_y_prob"])
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{label} (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Session-Based Test Set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES / "ieee_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(np.array(rf_metrics["confusion_matrix"]), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Random Forest Confusion Matrix (Test Set)")
    fig.savefig(FIGURES / "ieee_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    lc_scaler = StandardScaler()
    X_lc = lc_scaler.fit_transform(pd.concat([train_df, test_df])[full_features].fillna(0).values)
    y_lc = pd.concat([train_df, test_df])["is_anomaly"].astype(int).values
    sizes, tr, va = learning_curve(
        RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, class_weight="balanced", random_state=42),
        X_lc, y_lc, cv=3, train_sizes=np.linspace(0.2, 1.0, 5), scoring="roc_auc", n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sizes, tr.mean(axis=1), "o-", label="Train")
    ax.plot(sizes, va.mean(axis=1), "o-", label="Cross-val")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Random Forest Learning Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES / "ieee_learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "pc1": float(pca.explained_variance_ratio_[0]),
        "pc2": float(pca.explained_variance_ratio_[1]),
        "pc3": float(pca.explained_variance_ratio_[2]),
    }


def pack(m):
    return {k: round(m[k], 3) for k in ["roc_auc", "pr_auc", "precision", "recall", "f1", "train_time_s"] if k in m}


def main():
    train_df, val_df, test_df, labeled = prepare_dataset()
    total = len(train_df) + len(val_df) + len(test_df)
    full = feature_sets()["full"]
    rf = train_random_forest(train_df, val_df, test_df, full)
    xgbm = train_xgboost(train_df, val_df, test_df, full)
    ablations = run_ablations(train_df, val_df, test_df, full)
    pca_info = pca_motor_decomposition(train_df, test_df, full)
    pca_var = generate_figures(train_df, test_df, full, rf, xgbm)

    print("Training LSTM L=30 (session split)...")
    from session_eval import train_lstm_session_split
    if os.environ.get("SKIP_LSTM", "1") == "1":
        print("SKIP_LSTM=1; training L=30 only with 8 epochs...")
        lstm_l30 = train_lstm_session_split(labeled, sequence_length=30, epochs=8)
        lstm_sweep = {"30": pack(lstm_l30)}
        for L in [60, 120, 300]:
            m = train_lstm_session_split(labeled, sequence_length=L, epochs=5)
            lstm_sweep[str(L)] = pack(m)

    results = {
        "session_mapping": {str(i): {"folder": SESSION_ORDER[i - 1], "split": "train" if i in TRAIN_SESSION_IDS else "val" if i == VAL_SESSION_ID else "test"} for i in range(1, 9)},
        "split_counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df), "total": total},
        "anomaly_prevalence_pct": round(float(labeled["is_anomaly"].mean() * 100), 2),
        "sensor_statistics": sensor_statistics(labeled),
        "table_ii": {"random_forest": pack(rf), "xgboost": pack(xgbm), "lstm_L30": {k: lstm_l30[k] for k in ["roc_auc", "pr_auc", "precision", "recall", "f1", "train_time_s"]}},
        "ablations": ablations,
        "lstm_sweep": lstm_sweep,
        "pca_motor": pca_info,
        "pca_variance": pca_var,
        "rf_confusion_matrix": rf["confusion_matrix"],
        "rf_feature_importance": {k: round(v, 3) for k, v in rf["feature_importance"].items()},
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"table_ii": results["table_ii"], "split": results["split_counts"], "cm": rf["confusion_matrix"]}, indent=2))


if __name__ == "__main__":
    main()
