"""
E-commerce Customer & Sales Analytics
======================================
End-to-end analysis: EDA -> RFM segmentation -> churn prediction -> business
recommendations.

Run from the project root:  python notebooks/analysis.py
Outputs charts to ../visuals/ and prints key findings to stdout.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.dpi'] = 110

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
customers = pd.read_csv('../data/customers.csv', parse_dates=['signup_date'])
products = pd.read_csv('../data/products.csv')
orders = pd.read_csv('../data/orders.csv', parse_dates=['order_date'])
order_items = pd.read_csv('../data/order_items.csv')

ANALYSIS_DATE = orders['order_date'].max() + pd.Timedelta(days=1)

print("="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"Customers: {len(customers):,} | Products: {len(products):,} | "
      f"Orders: {len(orders):,} | Order lines: {len(order_items):,}")
print(f"Date range: {orders['order_date'].min().date()} -> {orders['order_date'].max().date()}")
print(f"Total revenue: ${orders['order_total'].sum():,.2f}")

# ------------------------------------------------------------------
# 2. MONTHLY REVENUE TREND
# ------------------------------------------------------------------
monthly = orders.copy()
monthly['month'] = monthly['order_date'].dt.to_period('M').astype(str)
monthly_rev = monthly.groupby('month').agg(
    revenue=('order_total', 'sum'),
    orders=('order_id', 'nunique'),
    active_customers=('customer_id', 'nunique')
).reset_index()

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(monthly_rev['month'], monthly_rev['revenue'], marker='o', linewidth=2, color='#2C6E91')
ax.set_title('Monthly Revenue Trend (2023-2024)', fontsize=13, fontweight='bold')
ax.set_ylabel('Revenue ($)')
ax.set_xlabel('Month')
plt.xticks(rotation=60, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig('../visuals/01_monthly_revenue.png')
plt.close()

# ------------------------------------------------------------------
# 3. CATEGORY PERFORMANCE
# ------------------------------------------------------------------
cat_perf = order_items.merge(products, on='product_id')
cat_summary = cat_perf.groupby('category').agg(
    revenue=('line_total', 'sum'),
    units=('quantity', 'sum')
).reset_index().sort_values('revenue', ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=cat_summary, x='revenue', y='category', ax=ax, palette='viridis')
ax.set_title('Revenue by Product Category', fontsize=13, fontweight='bold')
ax.set_xlabel('Revenue ($)')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig('../visuals/02_category_revenue.png')
plt.close()

print("\n" + "="*60)
print("CATEGORY PERFORMANCE")
print("="*60)
print(cat_summary.to_string(index=False))

# ------------------------------------------------------------------
# 4. RFM ANALYSIS (Recency, Frequency, Monetary)
# ------------------------------------------------------------------
rfm = orders.groupby('customer_id').agg(
    recency=('order_date', lambda x: (ANALYSIS_DATE - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('order_total', 'sum')
).reset_index()

# score 1(worst)-4(best) per metric using quartiles
rfm['R_score'] = pd.qcut(rfm['recency'], 4, labels=[4,3,2,1]).astype(int)
rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
rfm['M_score'] = pd.qcut(rfm['monetary'], 4, labels=[1,2,3,4]).astype(int)
rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

def segment_customer(row):
    if row['RFM_score'] >= 10:
        return 'Champions'
    elif row['RFM_score'] >= 8:
        return 'Loyal Customers'
    elif row['RFM_score'] >= 6:
        return 'Potential Loyalists'
    elif row['R_score'] <= 2 and row['F_score'] <= 2:
        return 'At Risk'
    else:
        return 'Needs Attention'

rfm['segment'] = rfm.apply(segment_customer, axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
seg_counts = rfm['segment'].value_counts()
colors = sns.color_palette('Set2', len(seg_counts))
ax.pie(seg_counts.values, labels=seg_counts.index, autopct='%1.0f%%',
       colors=colors, startangle=90, wedgeprops={'edgecolor':'white','linewidth':1.5})
ax.set_title('Customer Segments (RFM Analysis)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../visuals/03_rfm_segments.png')
plt.close()

print("\n" + "="*60)
print("RFM CUSTOMER SEGMENTS")
print("="*60)
print(seg_counts.to_string())
print(f"\nAverage monetary value by segment:")
print(rfm.groupby('segment')['monetary'].mean().round(2).sort_values(ascending=False).to_string())

# ------------------------------------------------------------------
# 5. CHURN LABEL + PREDICTIVE MODEL
#    Definition: churned = no order in the last 120 days of the dataset
# ------------------------------------------------------------------
CHURN_WINDOW_DAYS = 120
rfm['churned'] = (rfm['recency'] > CHURN_WINDOW_DAYS).astype(int)

# feature engineering
cust_features = customers.merge(rfm, on='customer_id', how='inner')
cust_features['tenure_days'] = (ANALYSIS_DATE - cust_features['signup_date']).dt.days
cust_features['avg_order_value'] = cust_features['monetary'] / cust_features['frequency']

features = ['frequency', 'monetary', 'avg_order_value', 'tenure_days']
X = cust_features[features]
y = cust_features['churned']

print("\n" + "="*60)
print(f"CHURN RATE: {y.mean()*100:.1f}% of customers (recency > {CHURN_WINDOW_DAYS} days)")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=['Active', 'Churned']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

coef_df = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_[0]
}).sort_values('coefficient')

fig, ax = plt.subplots(figsize=(7, 4))
colors_coef = ['#C0392B' if c > 0 else '#2C6E91' for c in coef_df['coefficient']]
ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors_coef)
ax.set_title('Churn Model — Feature Influence\n(red = increases churn risk)', fontsize=12, fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('../visuals/04_churn_feature_importance.png')
plt.close()

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Active','Churned'],
            yticklabels=['Active','Churned'], ax=ax)
ax.set_title('Confusion Matrix — Churn Prediction', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('../visuals/05_confusion_matrix.png')
plt.close()

# ------------------------------------------------------------------
# 6. COHORT RETENTION HEATMAP
# ------------------------------------------------------------------
orders_c = orders.copy()
orders_c['order_month'] = orders_c['order_date'].dt.to_period('M')
first_purchase = orders_c.groupby('customer_id')['order_month'].min().rename('cohort_month')
orders_c = orders_c.join(first_purchase, on='customer_id')
orders_c['cohort_index'] = (orders_c['order_month'] - orders_c['cohort_month']).apply(lambda x: x.n)

cohort_data = orders_c.groupby(['cohort_month', 'cohort_index'])['customer_id'].nunique().reset_index()
cohort_pivot = cohort_data.pivot(index='cohort_month', columns='cohort_index', values='customer_id')
cohort_size = cohort_pivot.iloc[:, 0]
retention = cohort_pivot.divide(cohort_size, axis=0).round(3)

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(retention.iloc[:12, :6], annot=True, fmt='.0%', cmap='YlGnBu', ax=ax, cbar_kws={'label':'Retention'})
ax.set_title('Monthly Cohort Retention (first 6 months)', fontsize=13, fontweight='bold')
ax.set_xlabel('Months Since First Purchase')
ax.set_ylabel('Signup Cohort')
plt.tight_layout()
plt.savefig('../visuals/06_cohort_retention.png')
plt.close()

# ------------------------------------------------------------------
# 7. EXPORT SUMMARY TABLES FOR THE DASHBOARD
# ------------------------------------------------------------------
monthly_rev.to_csv('../data/summary_monthly_revenue.csv', index=False)
cat_summary.to_csv('../data/summary_category.csv', index=False)
rfm[['customer_id','recency','frequency','monetary','segment','churned']].to_csv('../data/summary_rfm.csv', index=False)

print("\nAll visuals saved to /visuals, summary tables saved to /data.")
print("Done.")
