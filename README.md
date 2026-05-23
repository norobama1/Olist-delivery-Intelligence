# Delivery Intelligence — Olist E-Commerce

Predicting late deliveries on the Brazilian Olist marketplace. The goal is to understand *why* orders arrive late and build a model that flags high-risk orders before they ship.

**Dataset:** [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 110k+ delivered orders across 2016–2018.

---

## Project Structure

```
delivery-intelligence/
├── src/
│   ├── load_data.py                    # Merges raw Olist CSVs into a single table, creates delay target
│   └── feature.py                      # Engineers 11 features across temporal, physical, and logistics categories
├── notebooks/
│   └── EDA.ipynb                       # Exploratory analysis — 6 insights on delay patterns
├── dashboard/
│   ├── Dashboard.pbix                  # Power BI dashboard — delay by region, monthly trends, seller performance
│   └── dashboard_key_findings.docx     # Written summary of dashboard insights
├── models/                             # (Week 3) Trained model artefacts
└── app/                                # (Week 4) Prediction API
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

## Dashboard — Key Findings (`dashboard/Dashboard.pbix`)

**Dataset scope: 110,781 orders | Sep 2016 – Aug 2018 | Overall delay rate: 6.58%**

Full written summary: `dashboard/dashboard_key_findings.docx`

1. **Seasonal spikes are predictable and severe** — Delay rate hit 18% in Mar 2018 and 14% in Feb 2018, driven by Carnival and post-holiday volume surges. Pre-allocating carrier capacity for these windows is a direct lever.

2. **Northern states are systemically underserved** — Alagoas (21%) and Maranhão (18%) delay at 3× the national average despite low order volume. Last-mile coverage gaps in remote northern states require dedicated regional logistics partners, not just more volume.

3. **Premium orders delay more often** — Delay rate climbs from 5.5% for orders under R$50 to 8.8% above R$500. Bulkier, heavier items in high-value orders drive longer fulfilment times; priority handling could protect customer lifetime value.

4. **Mid-range shipping distances are the blind spot** — The 10–20 unit distance band peaks at 7.8% delay rate, then drops for longer hauls. Carrier networks appear to optimise for metro and long-haul lanes, leaving mid-range routes underserved.

5. **A small seller cohort drives outsized delays** — 12 sellers (with 50+ orders) exceed a 15% delay rate. Targeted SLA enforcement on this cohort is estimated to reduce overall delays by 0.3–0.5 percentage points.

6. **Estimates are padded by ~15 days** — Carriers deliver in 9 days on average against a 24-day estimate. Olist over-promises systematically, masking true operational performance. Tightening estimates by 5 days would improve transparency without increasing late deliveries.

---

## What's Next

- ✅ **Week 1:** Data pipeline, feature engineering, and EDA
- ✅ **Week 2:** Power BI dashboard — delay by region, monthly trends, seller performance, and order value vs delay
- **Week 3:** Train a binary classifier (XGBoost baseline), SHAP feature importance, replace zip proxy with haversine distance
- **Week 4:** FastAPI prediction endpoint, Streamlit dashboard for seller-level delay risk
