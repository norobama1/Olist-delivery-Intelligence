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
MODEL_PATH = ROOT / "models" / "model_bundle.joblib"
OUT_DIR    = ROOT / "models"

SHAP_SAMPLE = 2000


def load_test_split():
    """Reproduce the exact stratified test split used in training."""
    df = pd.read_csv(DATA_PATH, parse_dates=["order_purchase_timestamp"])
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    df = add_derived_features(df)
    _, X_test, _, y_test = train_test_split(
        df[ALL_FEATURES], df[TARGET], test_size=0.20, random_state=42, stratify=df[TARGET]
    )
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def feature_names():
    """Feature names in ColumnTransformer output order: num → pass → cat."""
    return NUMERIC_FEATURES + PASSTHROUGH_FEATURES + ORDINAL_CATEGORICAL_FEATURES


def extract_shap(explainer, X_arr, model):
    """Return (shap_values, expected_value) for the positive class.

    SHAP output varies by version:
      list of arrays      → old API, one array per class
      3D (n, f, classes)  → SHAP 0.46+, last axis is classes
      2D (n, f)           → already positive class
    """
    raw = explainer.shap_values(X_arr)
    ev  = explainer.expected_value
    if isinstance(raw, list):
        return raw[1], float(ev[1])
    if raw.ndim == 3:                    # (n_samples, n_features, n_classes)
        return raw[:, :, 1], float(ev[1])
    return raw, float(ev[1]) if hasattr(ev, "__len__") else float(ev)


def save_fig(filename):
    path = OUT_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def main():
    bundle     = joblib.load(MODEL_PATH)
    pipeline   = bundle["model"]
    threshold  = bundle["threshold"]
    model_name = bundle["model_name"]
    print(f"Model     : {model_name}")
    print(f"Threshold : {threshold:.3f}")

    # CalibratedClassifierCV wraps the pipeline — unwrap the first inner estimator.
    # SHAP values on the uncalibrated XGBoost are still valid for feature importance
    # since calibration only rescales probabilities monotonically.
    if hasattr(pipeline, "calibrated_classifiers_"):
        inner_pipe   = pipeline.calibrated_classifiers_[0].estimator
        preprocessor = inner_pipe.named_steps["pre"]
        clf          = inner_pipe.named_steps["clf"]
    else:
        preprocessor = pipeline.named_steps["pre"]
        clf          = pipeline.named_steps["clf"]

    X_test, y_test = load_test_split()

    # Transform to numpy — TreeExplainer requires array, not DataFrame
    X_arr = preprocessor.transform(X_test)
    names = feature_names()

    # Reproducible sample
    rng    = np.random.default_rng(42)
    idx    = rng.choice(len(X_arr), size=min(SHAP_SAMPLE, len(X_arr)), replace=False)
    X_shap = X_arr[idx]
    y_shap = y_test.iloc[idx].reset_index(drop=True)

    print(f"\nComputing SHAP on {len(X_shap):,} rows ...")
    explainer             = shap.TreeExplainer(clf)
    shap_values, exp_val  = extract_shap(explainer, X_shap, clf)

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
            exp_val,
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