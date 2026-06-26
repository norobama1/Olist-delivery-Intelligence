"""Pydantic models for the delivery-delay API."""
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# All 27 Brazilian federative units (UFs).
BR_STATES = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO",
}


class DeliveryFeatures(BaseModel):
    """Features known at purchase time for a single order.

    Matches the 15 columns the trained pipeline expects (is_weekend is derived
    in the API from purchase_dayofweek). Excludes identifiers, the target
    (delayed), and post-shipment leakage columns (shipping_days,
    delivery_delay_days).
    """

    estimated_days: float = Field(..., gt=0, le=365, description="Promised lead time: estimated delivery date minus purchase date, in days.")
    approval_hours: float = Field(..., ge=0, description="Hours from purchase to payment approval.")
    purchase_dayofweek: int = Field(..., ge=0, le=6, description="0=Monday ... 6=Sunday.")
    purchase_month: int = Field(..., ge=1, le=12)
    product_weight_g: float = Field(..., ge=0)
    product_volume_cm3: float = Field(..., ge=0)
    order_value: float = Field(..., gt=0)
    freight_value: float = Field(..., ge=0)
    zip_distance_proxy: float = Field(..., ge=0, description="Coarse seller->customer distance proxy from CEP prefixes.")
    seller_delay_rate: float = Field(..., ge=0, le=1, description="Seller's historical share of late deliveries, computed only from data before this order.")
    is_peak_delayed_period: int = Field(..., ge=0, le=1, description="1 if the order falls in a known high-delay month; set upstream in features.py.")
    n_items: int = Field(..., ge=1, description="Number of distinct items in the order.")
    seller_state: str = Field(..., description="Brazilian UF, e.g. 'SP'.")
    customer_state: str = Field(..., description="Brazilian UF, e.g. 'RJ'.")

    @field_validator("seller_state", "customer_state")
    @classmethod
    def _valid_uf(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in BR_STATES:
            raise ValueError(f"'{v}' is not a valid Brazilian state code")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "estimated_days": 6.0,
                "approval_hours": 0.4,
                "purchase_dayofweek": 4,
                "purchase_month": 11,
                "product_weight_g": 700.0,
                "product_volume_cm3": 6450.0,
                "order_value": 92.0,
                "freight_value": 16.25,
                "zip_distance_proxy": 24.0,
                "seller_delay_rate": 0.18,
                "is_peak_delayed_period": 1,
                "n_items": 2,
                "seller_state": "SP",
                "customer_state": "BA",
            }
        }
    }


class PredictionResponse(BaseModel):
    delay_probability: float = Field(..., description="P(delivered later than promised).")
    threshold: float = Field(..., description="Decision threshold tuned on the validation split.")
    predicted_delayed: bool
    risk_band: str = Field(..., description="high if p>=threshold, medium if p>=0.5*threshold, else low.")


class BatchRequest(BaseModel):
    orders: list[DeliveryFeatures]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_features: int
    threshold: Optional[float] = None