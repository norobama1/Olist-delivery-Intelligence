# Delivery Intelligence — Olist E-Commerce

Predicting late deliveries on the Brazilian Olist marketplace. The goal is to understand *why* orders arrive late and build a model that flags high-risk orders before they ship.

**Dataset:** [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 110k+ delivered orders across 2016–2018.

---

## Project Structure

```
delivery-intelligence/
├── src/
│   ├── load_data.py          # Merges raw Olist CSVs, creates delay target
│   ├── feature.py            # Engineers features across temporal, physical, and logistics categories
│   ├── model_config.py       # Feature lists, preprocessor, derived feature logic
│   ├── train_model.py        # LR / RF / XGBoost training, hyperparameter search, threshold optimisation
│   ├── predict_shap.py       # SHAP beeswarm, bar, dependence, and force plots
│   └── main.py               # FastAPI prediction server
├── app/
│   └── schemas.py            # Pydantic request/response models
├── tests/
│   └── test_api.py           # Pytest smoke tests (health, predict, batch, validation)
├── notebooks/
│   └── EDA.ipynb             # Exploratory analysis — 6 insights on delay patterns
├── dashboard/
│   ├── Dashboard.pbix        # Power BI dashboard
│   └── dashboard_key_findings.docx
├── models/                   # Trained artefacts and SHAP plots
├── Dockerfile
├── requirements.txt          # Full dev dependencies
└── requirements-api.txt      # Lean runtime dependencies for the API container
```

---

## Setup

Download the [Olist dataset from Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), place the CSVs in `data/`, then run:

```bash
pip install -r requirements.txt
python src/load_data.py
python src/feature.py
python src/train_model.py
```

---

## Pipeline

`load_data.py` merges 7 raw Olist tables, filters to delivered orders, and creates the target (`delayed = 1` if actual > estimated delivery date).

`feature.py` + `model_config.py` engineer 14 features across five categories:

| Category | Features |
|----------|----------|
| Temporal | `estimated_days`, `approval_hours`, `purchase_dayofweek`, `purchase_month` |
| Physical | `product_weight_g`, `product_volume_cm3`, `order_value`, `freight_value` |
| Logistics | `zip_distance_proxy`, `seller_delay_rate` ¹, `n_items` |
| Calendar | `is_peak_delayed_period`, `is_weekend` |
| Geographic | `seller_state`, `customer_state` |

¹ `seller_delay_rate` — each seller's historical delay rate computed from past orders only (`shift(1).expanding().mean()`), fully leak-free.

Output: `data/olist_processed.csv` — 110k rows, ready for modelling.

---

## Model Training (`src/train_model.py`)

Three models trained and compared on a stratified 70 / 15 / 15 train / val / test split:

| Model | Val PR-AUC | Selected |
|-------|-----------|----------|
| Logistic Regression | 0.1387 | |
| Random Forest | 0.2206 | |
| XGBoost (tuned + calibrated) | 0.2333 | ✅ |

**Selection criterion:** val PR-AUC on a held-out validation set the models never trained on.

**Threshold:** optimised for business cost (`5 × FN + 1 × FP`) — missing a delay costs 5× more than a false alarm — giving threshold = 0.118.

### Final model performance (XGBoost calibrated, stratified test set)

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.7875** |
| PR-AUC | **0.2577** |
| F1 | **0.3150** |
| Threshold | 0.118 |
| Delayed precision | 0.23 |
| Delayed recall | 0.51 |

---

## SHAP Explainability (`src/predict_shap.py`)

Top 3 features by mean |SHAP| (XGBoost):

1. `purchase_month` — seasonality is the strongest signal; March and November/December are high-delay periods
2. `estimated_days` — short promise windows are high-risk
3. `customer_state` — destination region drives logistics complexity

Plots saved to `models/`: beeswarm summary, bar importance, dependence plots (top 3), force plot (single delayed order).

---

## EDA — Key Findings (`notebooks/EDA.ipynb`)

**Overall delay rate: 6.6%**

1. **Delays are seasonal** — March (14.5%) and November (12%) are the two peak months, driven by post-Carnival and Black Friday surges.
2. **Zip distance proxy is weak** — the state-to-state heatmap gives a cleaner signal.
3. **Route matters more than distance** — DF→ES delays at 30.4%, DF→SC at 19.2%. Same-state routes (SP→SP) stay below 5%.
4. **Freight cost predicts delays, weight does not** — high-freight orders delay at 8.0% vs 5.2% for low-freight.
5. **Delays are concentrated in ~24% of sellers** — 321 of 1,344 active sellers cause 80% of all delays.
6. **Sellers who overpromise delay at 3.5× the rate** — orders with ≤7 day promised delivery delay at 16.5% vs 4.7% for 30+ day windows.

---

## Dashboard — Key Findings (`dashboard/Dashboard.pbix`)

**Dataset scope: 110,781 orders | Sep 2016 – Aug 2018 | Overall delay rate: 6.58%**

1. **Seasonal spikes are predictable** — Delay rate hit 18% in Mar 2018, 14% in Feb 2018.
2. **Northern states are systemically underserved** — Alagoas (21%) and Maranhão (18%) delay at 3× the national average.
3. **Premium orders delay more often** — Rate climbs from 5.5% under R$50 to 8.8% above R$500.
4. **Mid-range shipping distances are the blind spot** — The 10–20 unit distance band peaks at 7.8% delay rate.
5. **A small seller cohort drives outsized delays** — 12 sellers exceed 15% delay rate.
6. **Estimates are padded by ~15 days** — Carriers deliver in 9 days against a 24-day estimate.

---

## Prediction API (`src/main.py`)

FastAPI serving the best model with three endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model load status and feature count |
| `/predict` | POST | Single order delay probability + risk band |
| `/predict/batch` | POST | Batch predictions |

**Run locally:**
```bash
python src/main.py
# → http://localhost:8000/docs
```

**Run with Docker:**
```bash
docker build -t olist-delay-api .
docker run -p 8000:8000 olist-delay-api
```

**Example response:**
```json
{
  "delay_probability": 0.165,
  "threshold": 0.1477,
  "predicted_delayed": true,
  "risk_band": "high"
}
```

---

## Tests

```bash
pytest tests/
```

5 smoke tests covering health check, valid prediction, batch prediction, input validation (422 on bad state code), and high-risk order flagging.

---
