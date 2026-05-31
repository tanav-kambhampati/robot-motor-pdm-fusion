#!/usr/bin/env python3
"""Session-based model evaluation, ablations, and figure regeneration."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from session_eval import (
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
    save_results,
    sensor_statistics,
    train_lstm_session_split,
    train_random_forest,
    train_xgboost,
)

FIGURES_DIR = Path("figures")
RESULTS_PATH = Path("results/session_eval_results.json")


def _compact_row(name: str, metrics: dict) -> dict:
    return {
        "experiment": name,
        "roc_auc": round(metrics.get("roc_auc", float("nan")), 3),
        "pr_auc": round(metrics.get("pr_auc", float("nan")), 3),
        "precision": round(metrics.get("precision", float("nan")), 3),
        "recall": round(metrics.get("recall", float("nan")), 3),
        "f1": round(metrics.get("f1", float("nan")), 3),
        "train_time_s": round(metrics.get("train_time_s", 0.0), 1),
    }


def run_ablations(train_df, val_df, test_df, full_features):
    rows = []
    y_test = test_df["is_anomaly"].astype(int).values

    y_iqr = iqr_baseline_predict(test_df)
    rows.append(
        {
            "experiment": "IQR OR-rule (label rule as predictor)",
            "roc_auc": None,
            "pr_auc": None,
            "precision": round(float((y_iqr & y_test).sum() / max(y_iqr.sum(), 1)), 3),
            "recall": round(float((y_iqr & y_test).sum() / max(y_test.sum(), 1)), 3),
            "f1": round(
                float(2 * (y_iqr & y_test).sum() / max(y_iqr.sum() + y_test.sum(), 1)), 3
            ),
            "train_time_s": 0.0,
        }
    )

    iso_pred = isolation_forest_predict(train_df, test_df, full_features)
    iso_metrics = compute_metrics(y_test, iso_pred.astype(float), 0.5)
    rows.append({"experiment": "Isolation Forest", **{k: round(iso_metrics[k], 3) for k in ["roc_auc", "pr_auc", "precision", "recall", "f1"]}, "train_time_s": 0.0})

    ablation_map = {
        "RF — temperature only": feature_sets()["temperature_only"],
        "RF — voltage only": feature_sets()["voltage_only"],
        "RF — position only": feature_sets()["position_only"],
        "RF — raw channels only": feature_sets()["raw_only"],
        "RF — raw + rolling mean": feature_sets()["raw_plus_temp_roll"],
        "RF — full feature set": feature_sets()["full"],
    }

    rf_full = None
    for name, cols in ablation_map.items():
        metrics = train_random_forest(train_df, val_df, test_df, cols)
        row = _compact_row(name, metrics)
        rows.append(row)
        if name == "RF — full feature set":
            rf_full = metrics

    return rows, rf_full


def generate_figures(train_df, test_df, full_features, rf_metrics, xgb_metrics):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 14, "figure.facecolor": "white"})

    imp = pd.Series(rf_metrics["feature_importance"]).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    imp.plot(kind="barh", ax=ax, color="#2E86AB")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance (Session Split)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ieee_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    corr = train_df[full_features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ieee_correlation_heatmap.png", dpi=300, bbox_inches="tight")
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
    normal, anomaly = y_test == 0, y_test == 1
    ax1.scatter(test_pca[normal, 0], test_pca[normal, 1], test_pca[normal, 2], c="#A8DADC", s=8, alpha=0.5, label="Normal")
    ax1.scatter(test_pca[anomaly, 0], test_pca[anomaly, 1], test_pca[anomaly, 2], c="#E63946", s=12, alpha=0.7, label="Proxy anomaly")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax1.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    ax1.set_title("PCA by Anomaly Label")
    ax1.legend()

    ax2 = fig.add_subplot(122, projection="3d")
    for motor in sorted(set(motors)):
        mask = motors == motor
        ax2.scatter(test_pca[mask, 0], test_pca[mask, 1], test_pca[mask, 2], s=10, alpha=0.6, label=motor.replace("data_motor_", "M"))
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax2.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    ax2.set_title("PCA by Motor ID (Test Set)")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ieee_3d_pca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for label, metrics, color in [("Random Forest", rf_metrics, "#2E86AB"), ("XGBoost", xgb_metrics, "#F18F01")]:
        y = metrics["_y_test"]
        prob = metrics["_y_prob"]
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{label} (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Session-Based Test Set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ieee_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(np.array(rf_metrics["confusion_matrix"]), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Random Forest Confusion Matrix (Test Set)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ieee_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    lc_scaler = StandardScaler()
    X_lc = lc_scaler.fit_transform(
        pd.concat([train_df, test_df])[full_features].fillna(0).values
    )
    y_lc = pd.concat([train_df, test_df])["is_anomaly"].astype(int).values
    train_sizes, train_scores, val_scores = learning_curve(
        RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            class_weight="balanced", random_state=42,
        ),
        X_lc, y_lc, cv=3, train_sizes=np.linspace(0.2, 1.0, 5),
        scoring="roc_auc", n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train")
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Cross-val")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Random Forest Learning Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ieee_learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("SESSION-BASED EVALUATION AND ABLATIONS")
    print("=" * 70)

    train_df, val_df, test_df, labeled = prepare_dataset()
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"Train {len(train_df):,} ({100*len(train_df)/total:.1f}%)")
    print(f"Val   {len(val_df):,} ({100*len(val_df)/total:.1f}%)")
    print(f"Test  {len(test_df):,} ({100*len(test_df)/total:.1f}%)")
    print(f"Anomaly prevalence: {100*labeled['is_anomaly'].mean():.2f}%")

    full_features = feature_sets()["full"]
    stats = sensor_statistics(labeled)

    print("\nPrimary models...")
    rf_metrics = train_random_forest(train_df, val_df, test_df, full_features)
    xgb_metrics = train_xgboost(train_df, val_df, test_df, full_features)

    print("\nAblations...")
    ablation_rows, _ = run_ablations(train_df, val_df, test_df, full_features)

    print("\nLSTM sweep (may take several minutes)...")
    lstm_results = {}
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lstm-lengths", default="30,60,120,300")
    args, _ = parser.parse_known_args()
    lstm_lengths = [int(x) for x in args.lstm_lengths.split(",") if x.strip()]

    for seq_len in lstm_lengths:
        print(f"  L={seq_len}")
        lstm_results[str(seq_len)] = train_lstm_session_split(labeled, sequence_length=seq_len, epochs=20)

    pca_info = pca_motor_decomposition(train_df, test_df, full_features)

    def pack(m):
        return {k: round(m[k], 3) for k in ["roc_auc", "pr_auc", "precision", "recall", "f1", "train_time_s"] if k in m}

    results = {
        "session_mapping": {
            str(i): {
                "folder": SESSION_ORDER[i - 1],
                "split": "train" if i in TRAIN_SESSION_IDS else "val" if i == VAL_SESSION_ID else "test",
            }
            for i in range(1, 9)
        },
        "split_counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df), "total": total},
        "anomaly_prevalence_pct": round(float(labeled["is_anomaly"].mean() * 100), 2),
        "sensor_statistics": stats,
        "table_ii": {
            "random_forest": pack(rf_metrics),
            "xgboost": pack(xgb_metrics),
            "lstm_L30": pack(lstm_results["30"]),
        },
        "ablations": ablation_rows,
        "lstm_sweep": {k: pack(v) for k, v in lstm_results.items()},
        "pca_motor": pca_info,
        "rf_confusion_matrix": rf_metrics["confusion_matrix"],
        "rf_feature_importance": {k: round(v, 3) for k, v in rf_metrics["feature_importance"].items()},
    }

    save_results(results, str(RESULTS_PATH))
    generate_figures(train_df, test_df, full_features, rf_metrics, xgb_metrics)

    print(f"\nSaved {RESULTS_PATH}")
    for name, m in [("RF", rf_metrics), ("XGB", xgb_metrics), ("LSTM30", lstm_results["30"])]:
        print(f"{name}: ROC={m['roc_auc']:.3f} PR={m['pr_auc']:.3f} F1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
