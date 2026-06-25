"""FastAPI serving layer for the Olist delivery-delay model.

Loads a model bundle (estimator + tuned threshold + feature list) written by
src/train.py and exposes health and prediction endpoints. The API is
model-agnostic: it serves whichever estimator the training registry selected,
and reads the threshold from the bundle rather than hardcoding it.
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to path so `app/` is importable from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (
    BatchRequest,
    BatchResponse,
    DeliveryFeatures,
    HealthResponse,
    PredictionResponse,
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model_bundle.joblib"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model bundle not found at {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    state["model"] = bundle["model"]
    state["threshold"] = float(bundle["threshold"])
    state["feature_names"] = list(bundle["feature_names"])
    yield
    state.clear()


app = FastAPI(
    title="Olist Delivery Delay API",
    description="Predicts the probability that an order is delivered later than promised.",
    version="1.0.0",
    lifespan=lifespan,
)


def _build_frame(records: list[dict]) -> pd.DataFrame:
    """Turn request records into the exact feature matrix the model expects."""
    df = pd.DataFrame(records)
    # is_weekend is derived at train time (model_config.py); mirror it here.
    df["is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)
    missing = set(state["feature_names"]) - set(df.columns)
    if missing:
        raise HTTPException(422, f"Cannot build features, missing: {sorted(missing)}")
    return df[state["feature_names"]]


def _score(df: pd.DataFrame) -> list[dict]:
    try:
        proba = state["model"].predict_proba(df)[:, 1]
    except Exception as exc:  # surfaces train/serve skew (e.g. unencoded inputs)
        raise HTTPException(500, f"Model scoring failed: {exc}")
    t = state["threshold"]
    results = []
    for p in proba:
        p = float(p)
        band = "high" if p >= t else "medium" if p >= 0.5 * t else "low"
        results.append(
            {
                "delay_probability": round(p, 4),
                "threshold": t,
                "predicted_delayed": p >= t,
                "risk_band": band,
            }
        )
    return results


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = "model" in state
    return HealthResponse(
        status="ok" if loaded else "model_not_loaded",
        model_loaded=loaded,
        n_features=len(state.get("feature_names", [])),
        threshold=state.get("threshold"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: DeliveryFeatures):
    result = _score(_build_frame([features.model_dump()]))[0]
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    records = [order.model_dump() for order in req.orders]
    results = _score(_build_frame(records))
    return BatchResponse(predictions=[PredictionResponse(**r) for r in results])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)