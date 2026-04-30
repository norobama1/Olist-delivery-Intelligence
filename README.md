# Delivery Intelligence — Olist E-Commerce

Predicting late deliveries on the Brazilian Olist marketplace. The goal is to understand *why* orders arrive late and build a model that flags high-risk orders before they ship.

**Dataset:** [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 110k+ delivered orders across 2016–2018.

---

## Project Structure

```
delivery-intelligence/
├── src/
│   ├── load_data.py      # Merges raw Olist CSVs into a single table, creates delay target
│   └── feature.py        # Engineers 11 features across temporal, physical, and logistics categories
├── notebooks/
│   └── EDA.ipynb         # Exploratory analysis — 6 insights on delay patterns
├── models/               # (Week 3) Trained model artefacts
├── app/                  # (Week 4) Prediction API
└── dashboard/            # (Week 4) Streamlit dashboard
```

---

## Setup

Download the [Olist dataset from Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), place the CSVs in `data/`, then run:

```bash
pip install -r requirements.txt
python src/load_data.py
python src/features.py
```

---

## Pipeline

`load_data.py` merges 7 raw Olist tables, filters to delivered orders, and creates the target (`delayed = 1` if actual > estimated delivery date).

`feature.py` engineers 11 features across four categories:

| Category | Features |
|----------|----------|
| Temporal | `shipping_days`, `estimated_days`, `approval_hours`, `purchase_dayofweek`, `purchase_month` |
| Physical | `product_weight_g`, `product_volume_cm3`, `order_value` |
| Logistics | `freight_value`, `zip_distance_proxy` |
| Calendar | `is_peak_delayed_period` (flags March, November, December) |

Output: `data/olist_processed.csv` — 110k rows, ready for modelling.

---

## EDA — Key Findings (`notebooks/EDA.ipynb`)

**Overall delay rate: 6.6%**

1. **Delays are seasonal** — March (14.5%) and November (12%) are the two peak months, driven by post-Carnival and Black Friday surges. Mid-year months run as low as 2%.

2. **Zip distance proxy is weak** — the 2-digit zip prefix difference doesn't map cleanly to geography. The state-to-state heatmap gives a cleaner signal.

3. **Route matters more than distance** — DF→ES delays at 30.4%, DF→SC at 19.2%. Same-state routes (SP→SP) stay below 5%. The problem is specific corridors, not distance in general.

4. **Freight cost predicts delays, weight does not** — high-freight orders delay at 8.0% vs 5.2% for low-freight. Freight cost already encodes weight, distance and route difficulty, so weight adds nothing on top.

5. **Delays are concentrated in ~24% of sellers** — 321 of 1,344 active sellers cause 80% of all delays. This is a seller problem, not a platform-wide logistics problem.

6. **Sellers who overpromise delay at 3.5x the rate** — orders with ≤7 day promised delivery delay at 16.5% vs 4.7% for 30+ day windows. Constraining aggressive delivery estimates would reduce delays without touching logistics.

---

## What's Next

- **Week 2:** Power BI dashboard — delay by region, monthly trends, seller performance, and order value vs delay
- **Week 3:** Train a binary classifier (XGBoost baseline), SHAP feature importance, replace zip proxy with haversine distance
- **Week 4:** FastAPI prediction endpoint, Streamlit dashboard for seller-level delay risk
