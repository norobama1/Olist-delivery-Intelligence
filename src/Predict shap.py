import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from model_config import (
    ALL_FEATURES, TARGET,
    NUMERIC_FEATURES, PASSTHROUGH_FEATURES, ORDINAL_CATEGORICAL_FEATURES,
    add_derived_features,
)

ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data" / "olist_processed.csv"
MODEL_PATH = ROOT / "models" / "xgb_pipeline.joblib"
META_PATH  = ROOT / "models" / "model_meta.joblib"
OUT_DIR    = ROOT / "models"

# Rows passed to SHAP — 2 000 balances speed and representativeness
SHAP_SAMPLE = 2000


def load_test_split():
    """Reproduce the exact stratified test split used in training."""
    df = pd.read_csv(DATA_PATH, parse_dates=["order_purchase_timestamp"])
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    df = add_derived_features(df)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def feature_names():
    """Feature names in ColumnTransformer output order: num → pass → cat."""
    return NUMERIC_FEATURES + PASSTHROUGH_FEATURES + ORDINAL_CATEGORICAL_FEATURES


def save_fig(filename):
    path = OUT_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def main():
    pipeline  = joblib.load(MODEL_PATH)
    threshold = joblib.load(META_PATH)["threshold"]
    print(f"Loaded model  : {MODEL_PATH.name}")
    print(f"Threshold     : {threshold:.3f}")

    preprocessor = pipeline.named_steps["pre"]
    xgb_model    = pipeline.named_steps["clf"]

    X_test, y_test = load_test_split()

    # Transform to numpy array — TreeExplainer requires raw array, not DataFrame
    X_arr = preprocessor.transform(X_test)
    names = feature_names()

    # Reproducible sample
    rng    = np.random.default_rng(42)
    idx    = rng.choice(len(X_arr), size=min(SHAP_SAMPLE, len(X_arr)), replace=False)
    X_shap = X_arr[idx]
    y_shap = y_test.iloc[idx].reset_index(drop=True)

    print(f"\nComputing SHAP on {len(X_shap):,} rows ...")
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_shap)

    # ── 1. Beeswarm summary ───────────────────────────────────────────────────
    shap.summary_plot(
        shap_values, X_shap,
        feature_names=names,
        max_display=12,
        show=False,
    )
    plt.title("SHAP Feature Impact — Delivery Delay Prediction", pad=14)
    save_fig("shap_summary.png")

    # ── 2. Bar chart — mean |SHAP| importance ────────────────────────────────
    shap.summary_plot(
        shap_values, X_shap,
        feature_names=names,
        plot_type="bar",
        max_display=12,
        show=False,
    )
    plt.title("Mean |SHAP| — Feature Importance", pad=14)
    save_fig("shap_importance.png")

    # ── 3. Dependence plots — top 3 features by mean |SHAP| ──────────────────
    mean_abs = np.abs(shap_values).mean(axis=0)
    top3     = np.argsort(mean_abs)[::-1][:3]

    for rank, feat_idx in enumerate(top3):
        fname = names[feat_idx]
        shap.dependence_plot(
            feat_idx, shap_values, X_shap,
            feature_names=names,
            show=False,
        )
        plt.title(f"SHAP Dependence — {fname}", pad=14)
        save_fig(f"shap_dep_{rank + 1}_{fname}.png")

    # ── 4. Force plot — single delayed order ─────────────────────────────────
    delayed_positions = y_shap[y_shap == 1].index.tolist()
    if delayed_positions:
        i = delayed_positions[0]
        shap.force_plot(
            explainer.expected_value,
            shap_values[i],
            X_shap[i],
            feature_names=names,
            matplotlib=True,
            show=False,
        )
        plt.title("SHAP Force Plot — Single Delayed Order", pad=20)
        save_fig("shap_force_example.png")
    else:
        print("No delayed orders in SHAP sample — increase SHAP_SAMPLE to generate force plot.")

    print("\nDone. All plots saved to models/")


if __name__ == "__main__":
    main()