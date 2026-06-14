import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from model_config import ALL_FEATURES, TARGET, add_derived_features, build_preprocessor

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "olist_processed.csv"
MODEL_DIR = ROOT / "models"


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["order_purchase_timestamp"])
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    df = add_derived_features(df)
    return df[ALL_FEATURES], df[TARGET]


# Missing a delay (customer complaint) costs 5× more than a false alarm.
_COST_FN, _COST_FP = 5, 1

def optimal_threshold(y_true, y_prob, cost_fn=_COST_FN, cost_fp=_COST_FP):
    """Threshold that minimises cost_fn × FN + cost_fp × FP on the given split."""
    y_arr = np.asarray(y_true)
    best_cost, best_t = np.inf, 0.5
    for t in np.linspace(0.01, 0.99, 300):
        pred = (y_prob >= t).astype(int)
        fn = ((y_arr == 1) & (pred == 0)).sum()
        fp = ((y_arr == 0) & (pred == 1)).sum()
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_cost, best_t = cost, t
    return float(best_t)


def plot_pr_curve(y_true, y_prob, threshold, path):
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    idx = int(np.argmin(np.abs(thresholds - threshold)))
    no_skill = float(np.asarray(y_true).mean())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, lw=2, label=f"XGBoost  AP={ap:.3f}")
    ax.scatter(rec[idx], prec[idx], s=120, zorder=5,
               label=f"Operating point  t={threshold:.3f}  "
                     f"(FN cost={_COST_FN}×, FP cost={_COST_FP}×)")
    ax.axhline(no_skill, linestyle="--", color="grey",
               label=f"No-skill baseline  {no_skill:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Delivery Delay Prediction")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def evaluate(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    bar = "─" * max(1, 48 - len(name))
    print(f"\n── {name} {bar}")
    print(f"  PR-AUC  : {average_precision_score(y_true, y_prob):.4f}")
    print(f"  ROC-AUC : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  F1      : {f1_score(y_true, y_pred):.4f}  (threshold={threshold:.3f})")
    print(classification_report(
        y_true, y_pred,
        target_names=["on-time", "delayed"],
        digits=3,
    ))


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    X, y = load_data()

    # ── Stratified 80/20 train/test split ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Val carved from train (threshold tuning + early stopping only) ─────────
    # Test set is never touched until the evaluate() calls at the bottom.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = neg / pos
    print(f"Tr   : {len(y_tr):,}  |  delayed={pos:,} ({pos/len(y_tr)*100:.2f}%)")
    print(f"Val  : {len(y_val):,}")
    print(f"Test : {len(y_test):,}")
    print(f"scale_pos_weight : {spw:.2f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Registry: populated after each model — selection happens at the end
    registry = {}

    # ── 1. Logistic Regression ────────────────────────────────────────────────
    lr_search = RandomizedSearchCV(
        Pipeline([
            ("pre", build_preprocessor()),
            ("sclaer",StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                solver = "saga",
                max_iter=5000,
                random_state=42,
            )),
        ]),
        {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
        n_iter=8,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    lr_search.fit(X_tr, y_tr)
    lr_best   = lr_search.best_estimator_
    lr_val_prob = lr_best.predict_proba(X_val)[:, 1]
    lr_thresh  = optimal_threshold(y_val, lr_val_prob)
    evaluate("Logistic Regression", y_test, lr_best.predict_proba(X_test)[:, 1], lr_thresh)
    registry["Logistic Regression"] = {
        "pipeline":  lr_best,
        "threshold": lr_thresh,
        "val_prauc": average_precision_score(y_val, lr_val_prob),
    }
    
    # ── 2. Random Forest ──────────────────────────────────────────────────────
    rf_search = RandomizedSearchCV(
        Pipeline([
            ("pre", build_preprocessor()),
            ("clf", RandomForestClassifier(
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        {
            "clf__n_estimators":     [200, 300, 500],
            "clf__max_depth":        [8, 10, 15, None],
            "clf__min_samples_leaf": [1, 5, 10],
            "clf__max_features":     ["sqrt", 0.5],
        },
        n_iter=10,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    rf_search.fit(X_tr, y_tr)
    rf_best   = rf_search.best_estimator_
    rf_val_prob = rf_best.predict_proba(X_val)[:, 1]
    rf_thresh  = optimal_threshold(y_val, rf_val_prob)
    evaluate("Random Forest", y_test, rf_best.predict_proba(X_test)[:, 1], rf_thresh)
    registry["Random Forest"] = {
        "pipeline":  rf_best,
        "threshold": rf_thresh,
        "val_prauc": average_precision_score(y_val, rf_val_prob),
    }

    # ── 3. XGBoost — three stages ─────────────────────────────────────────────
    # Stage 1: RandomizedSearchCV finds best structural params.
    # n_estimators fixed at 300; no early stopping during CV.
    xgb_search = RandomizedSearchCV(
        Pipeline([
            ("pre", build_preprocessor()),
            ("clf", XGBClassifier(
                n_estimators=300,
                scale_pos_weight=spw,
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        {
            "clf__max_depth":        [3, 4, 5, 6],
            "clf__learning_rate":    [0.01, 0.05, 0.1],
            "clf__subsample":        [0.7, 0.8, 0.9],
            "clf__colsample_bytree": [0.6, 0.8, 1.0],
            "clf__min_child_weight": [3, 5, 10],
        },
        n_iter=30,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=1,
    )
    xgb_search.fit(X_tr, y_tr)
    best_params = {
        k.replace("clf__", ""): v
        for k, v in xgb_search.best_params_.items()
    }
    print(f"\nBest params : {best_params}")

    # Stage 2: early stopping on val set to find optimal n_estimators.
    # Preprocessor fitted on X_tr only.
    pre_es = build_preprocessor()
    X_tr_arr  = pre_es.fit_transform(X_tr)
    X_val_arr = pre_es.transform(X_val)

    xgb_es = XGBClassifier(
        **best_params,
        n_estimators=1000,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    xgb_es.fit(X_tr_arr, y_tr, eval_set=[(X_val_arr, y_val)], verbose=False)
    best_n = xgb_es.best_iteration
    print(f"Early stopping : best iteration = {best_n}")

    # Threshold found on val — only use here, never on test.
    xgb_val_prob = xgb_es.predict_proba(X_val_arr)[:, 1]
    xgb_thresh   = optimal_threshold(y_val, xgb_val_prob)

    # Stage 3: calibrated refit on full X_train with best_n.
    # Calibration: 3-fold CV isotonic regression on X_train aligns probs to true rates.
    # cv="prefit" removed in sklearn 1.8 — pass an unfitted clone with same params.
    cal_pipe = CalibratedClassifierCV(
        Pipeline([
            ("pre", build_preprocessor()),
            ("clf", XGBClassifier(
                **best_params,
                n_estimators=best_n,
                scale_pos_weight=spw,
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        cv=3,
        method="isotonic",
    )
    cal_pipe.fit(X_train, y_train)
    xgb_pipeline = cal_pipe

    xgb_val_prob_cal = xgb_pipeline.predict_proba(X_val)[:, 1]
    xgb_thresh       = optimal_threshold(y_val, xgb_val_prob_cal)

    xgb_test_prob = xgb_pipeline.predict_proba(X_test)[:, 1]
    evaluate("XGBoost (calibrated)", y_test, xgb_test_prob, xgb_thresh)
    plot_pr_curve(y_test, xgb_test_prob, xgb_thresh, MODEL_DIR / "pr_curve.png")

    registry["XGBoost"] = {
        "pipeline":  xgb_pipeline,
        "threshold": xgb_thresh,
        "val_prauc": average_precision_score(y_val, xgb_val_prob),  # Stage 2: trained on X_tr, val is unseen
    }

     # ── Auto-select best model by val PR-AUC ──────────────────────────────────
    best_name = max(registry, key=lambda k: registry[k]["val_prauc"])
    best      = registry[best_name]
 
    print("\n── Model selection ─────────────────────────────────")
    for name, entry in registry.items():
        marker = " ← selected" if name == best_name else ""
        print(f"  {name:<25} val PR-AUC = {entry['val_prauc']:.4f}{marker}")
 
    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(
        {
            "model":         best["pipeline"],
            "threshold":     best["threshold"],
            "model_name":    best_name,
            "feature_names": list(X_train.columns),
        },
        MODEL_DIR / "model_bundle.joblib",
    )
    print(f"\nSaved → {MODEL_DIR / 'model_bundle.joblib'}  ({best_name})")


if __name__ == "__main__":
    main()