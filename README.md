# Ledgerline — E-commerce Customer & Sales Analytics

An end-to-end analytics project on a mid-size e-commerce business: revenue trends,
product performance, customer segmentation (RFM), cohort retention, and a churn
prediction model — packaged the way a stakeholder-facing analysis would be delivered.

**Stack:** Python (pandas, scikit-learn, matplotlib/seaborn) · SQL (SQLite) · HTML/Chart.js dashboard

---

## 1. Business problem

A retail company selling furniture, electronics, office supplies, and apparel wants to
know three things:

1. Where is revenue actually coming from (products, seasons, customer segments)?
2. Which customers are valuable, and which are quietly slipping away?
3. Can we predict churn early enough to act on it?

## 2. Data

Since no proprietary dataset was available, a realistic synthetic dataset was generated
(`generate_data.py`) rather than reusing an overused public dataset (e.g., Titanic/Iris).
It mimics real transactional messiness: uneven customer activity, seasonal holiday spikes,
variable discounting, and multi-item orders.

| Table | Rows | Description |
|---|---|---|
| `customers.csv` | 1,000 | region, segment, signup date |
| `products.csv` | 22 | category, unit cost, unit price |
| `orders.csv` | 6,600 | order-level date and total |
| `order_items.csv` | 12,393 | line items with quantity/discount |

Date range: **Jan 2023 – Dec 2024**. Total revenue: **$1.34M**.

## 3. Approach

```
generate_data.py  →  data/*.csv  →  SQLite (data/ecommerce.db)
                                  →  sql/business_questions.sql   (10 business queries)
                                  →  notebooks/analysis.py        (EDA + RFM + churn model)
                                  →  visuals/*.png                (charts for reporting)
                                  →  dashboard/dashboard.html     (interactive summary)
```

**SQL (`sql/business_questions.sql`)** — 10 queries answering real stakeholder questions:
monthly revenue trend, top customers by lifetime value, category profit margins,
best-sellers, AOV by segment/region, recency-based churn candidates, cohort retention,
discount-vs-basket-size, repeat purchase rate, and seasonality.

**Python analysis (`notebooks/analysis.py`)**:
- Exploratory analysis of revenue, category mix, and seasonality
- **RFM segmentation** — customers scored on Recency/Frequency/Monetary and grouped into
  Champions, Loyal Customers, Potential Loyalists, At Risk, and Needs Attention
- **Cohort retention** heatmap by signup month
- **Churn prediction** — logistic regression (customer churned = no order in 120+ days),
  evaluated with a train/test split, classification report, ROC-AUC, and a confusion matrix

**Dashboard (`dashboard/dashboard.html`)** — a standalone interactive summary (Chart.js)
covering KPIs, monthly revenue, segment mix, category revenue, and order/customer trend.
Open it directly in any browser — no server needed.

## 4. Key findings

- **Revenue is concentrated**: Furniture is 63% of revenue from just 32% of units sold —
  it's a high-price, lower-volume category and the one worth protecting on margin.
- **December drives outsized revenue** (~2x a normal month) via a holiday ordering spike —
  useful for inventory and staffing planning.
- **87% repeat purchase rate** overall, but customer value is highly uneven: "Champions"
  (30% of customers) average **$3,254** lifetime value vs. **$307** for "At Risk" customers.
- **30% of customers are churned** under a 120-day inactivity definition.
- The churn model reaches **ROC-AUC 0.81** using only frequency, monetary value, average
  order value, and tenure — no need for complex features to get a usable early-warning signal.
  Recency-adjacent features (low frequency, low monetary value) are the strongest churn drivers.

## 5. Recommendations

1. **Prioritize retention offers for "At Risk" customers with historically high AOV** — they
   already showed high willingness to pay; a small win-back incentive has a better ROI than
   discounting broadly.
2. **Plan inventory and staffing around the December spike** rather than treating each month
   as equal — the seasonal lift is consistent across both years in the dataset.
3. **Investigate why Office Supplies has the highest unit volume but lowest revenue share** —
   possible bundling or upsell opportunity attaching it to Furniture/Electronics orders.
4. **Operationalize the churn model as a monthly scoring job**, flagging customers who cross
   the risk threshold before they hit 120 days of inactivity, not after.

## 6. Repo structure

```
├── data/                     # raw + summary CSVs, SQLite DB
├── notebooks/analysis.py     # main analysis script
├── sql/business_questions.sql
├── visuals/                  # exported PNG charts
├── dashboard/dashboard.html  # interactive dashboard
├── generate_data.py          # synthetic data generator
└── README.md
```

## 7. How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python generate_data.py
cd notebooks && python analysis.py
# open dashboard/dashboard.html in a browser
```

---

*Dataset is synthetic and generated for portfolio/demonstration purposes. Methodology
(RFM, cohort analysis, churn modeling) generalizes directly to real transactional data.*
