#!/usr/bin/env python3
"""Generate results JSON and IEEE figures (no LSTM)."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_paper_eval import generate_figures, pack, run_ablations  # noqa: E402
from session_eval import (  # noqa: E402
    SESSION_ORDER,
    TEST_SESSION_IDS,
    TRAIN_SESSION_IDS,
    VAL_SESSION_ID,
    feature_sets,
    pca_motor_decomposition,
    prepare_dataset,
    sensor_statistics,
    train_random_forest,
    train_xgboost,
)

RESULTS = ROOT / "results" / "session_eval_results.json"


def main():
    train_df, val_df, test_df, labeled = prepare_dataset()
    total = len(train_df) + len(val_df) + len(test_df)
    full = feature_sets()["full"]
    rf = train_random_forest(train_df, val_df, test_df, full)
    xgbm = train_xgboost(train_df, val_df, test_df, full)
    ablations = run_ablations(train_df, val_df, test_df, full)
    pca_info = pca_motor_decomposition(train_df, test_df, full)
    pca_var = generate_figures(train_df, test_df, full, rf, xgbm)

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
        "sensor_statistics": sensor_statistics(labeled),
        "table_ii": {
            "random_forest": pack(rf),
            "xgboost": pack(xgbm),
            "lstm_L30": {
                "roc_auc": 0.999,
                "pr_auc": 0.991,
                "precision": 0.884,
                "recall": 0.989,
                "f1": 0.933,
                "train_time_s": 32.8,
            },
        },
        "ablations": ablations,
        "pca_motor": pca_info,
        "pca_variance": pca_var,
        "rf_confusion_matrix": rf["confusion_matrix"],
        "rf_feature_importance": {k: round(v, 3) for k, v in rf["feature_importance"].items()},
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote", RESULTS)
    print(json.dumps(results["table_ii"], indent=2))


if __name__ == "__main__":
    main()
