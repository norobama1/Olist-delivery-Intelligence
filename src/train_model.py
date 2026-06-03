import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             f1_score, precision_recall_curve, roc_auc_score)
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


def optimal_threshold(y_true, y_prob):
    """Find threshold that maximises F1 on the given split."""
    prec, rec, threshold = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    return float(threshold[f1.argmax()])


def evaluate(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    bar = "-" * max(1, 48 - len(name))
    print(f"\n── {name} {bar}")
    print(f"  PR-AUC  : {average_precision_score(y_true, y_prob):.4f}")
    print(f"  ROC-AUC : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  F1      : {f1_score(y_true, y_pred):.4f}  (threshold={threshold:.3f})")
    print(classification_report(y_true, y_pred, target_names=["on-time", "delayed"], digits=3))


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = neg / pos

    print(f"Train  : {len(y_tr):,} rows  |  delayed={pos:,} ({pos/len(y_tr)*100:.2f}%)")
    print(f"Val    : {len(y_val):,} rows")
    print(f"Test   : {len(y_test):,} rows")
    print(f"scale_pos_weight : {spw:.2f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── 1. Logistic Regression (baseline) ────────────────────────────────────
    lr_pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(class_weight="balanced", solver="saga",
                                   max_iter=2000, random_state=42)),
    ])
    lr_pipe.fit(X_tr, y_tr)
    lr_thresh = optimal_threshold(y_val, lr_pipe.predict_proba(X_val)[:, 1])
    evaluate("Logistic Regression", y_test, lr_pipe.predict_proba(X_test)[:, 1], lr_thresh)

    # ── 2. Random Forest + hyperparameter search ──────────────────────────────
    rf_pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)),
    ])
    rf_param_dist = {
        "clf__n_estimators":   [200, 300, 500],
        "clf__max_depth":      [6, 8, 10, None],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__max_features":   ["sqrt", 0.5],
    }
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_param_dist, n_iter=15, scoring="average_precision",
        cv=cv, random_state=42, n_jobs=1, verbose=1,  # n_jobs=1: RF handles its own parallelism
    )
    rf_search.fit(X_tr, y_tr)
    rf_best = rf_search.best_estimator_
    rf_thresh = optimal_threshold(y_val, rf_best.predict_proba(X_val)[:, 1])
    print(f"\nRF best params : {rf_search.best_params_}")
    evaluate("Random Forest", y_test, rf_best.predict_proba(X_test)[:, 1], rf_thresh)

    # ── 3. XGBoost — search then refit with early stopping ───────────────────
    xgb_pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", XGBClassifier(
            scale_pos_weight=spw,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    param_dist = {
        "clf__max_depth":         [3, 4, 5, 6],
        "clf__learning_rate":     [0.01, 0.05, 0.1],
        "clf__subsample":         [0.7, 0.8, 0.9],
        "clf__colsample_bytree":  [0.6, 0.8, 1.0],
        "clf__min_child_weight":  [3, 5, 10],
    }
    search = RandomizedSearchCV(
        xgb_pipe, param_dist, n_iter=30, scoring="average_precision",
        cv=cv, random_state=42, n_jobs=1, verbose=1,  # n_jobs=1: XGB handles its own parallelism
    )
    search.fit(X_tr, y_tr)
    print(f"\nBest params : {search.best_params_}")

    # Refit best params with early stopping — preprocess val set manually
    best_clf_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    pre = build_preprocessor()
    X_tr_t  = pre.fit_transform(X_tr)
    X_val_t = pre.transform(X_val)

    xgb_clf = XGBClassifier(
        **best_clf_params,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    xgb_clf.fit(X_tr_t, y_tr, eval_set=[(X_val_t, y_val)], verbose=False)
    print(f"XGBoost stopped at iteration {xgb_clf.best_iteration}")

    # Wrap fitted pre + clf into a pipeline for a consistent predict interface
    xgb_best = Pipeline([("pre", pre), ("clf", xgb_clf)])
    xgb_thresh = optimal_threshold(y_val, xgb_best.predict_proba(X_val)[:, 1])
    evaluate("XGBoost (tuned)", y_test, xgb_best.predict_proba(X_test)[:, 1], xgb_thresh)

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(xgb_best, MODEL_DIR / "xgb_pipeline.joblib")
    joblib.dump(
        {"threshold": xgb_thresh, "scale_pos_weight": spw},
        MODEL_DIR / "model_meta.joblib",
    )
    print(f"\nSaved → {MODEL_DIR / 'xgb_pipeline.joblib'}")
    print(f"Saved → {MODEL_DIR / 'model_meta.joblib'}")


if __name__ == "__main__":
    main()
