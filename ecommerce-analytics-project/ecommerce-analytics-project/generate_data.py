import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ---------- Customers ----------
N_CUSTOMERS = 1000
regions = ['North', 'South', 'East', 'West', 'Central']
segments = ['Consumer', 'Corporate', 'Home Office']
signup_start = datetime(2022, 1, 1)
signup_end = datetime(2023, 12, 31)

customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, N_CUSTOMERS+1)]
signup_dates = [signup_start + timedelta(days=np.random.randint(0, (signup_end-signup_start).days)) for _ in range(N_CUSTOMERS)]

customers = pd.DataFrame({
    'customer_id': customer_ids,
    'region': np.random.choice(regions, N_CUSTOMERS, p=[0.22,0.2,0.18,0.25,0.15]),
    'segment': np.random.choice(segments, N_CUSTOMERS, p=[0.55,0.3,0.15]),
    'signup_date': signup_dates
})

# ---------- Products ----------
categories = {
    'Electronics': ['Wireless Mouse','Bluetooth Speaker','USB-C Hub','Laptop Stand','Webcam','Power Bank'],
    'Furniture': ['Office Chair','Standing Desk','Bookshelf','Desk Lamp','Filing Cabinet'],
    'Office Supplies': ['Notebook Pack','Sticky Notes','Stapler','Pen Set','Whiteboard','Binder Clips'],
    'Apparel': ['T-Shirt','Hoodie','Cap','Backpack','Water Bottle']
}
products = []
pid = 1
base_price = {'Electronics':(15,120),'Furniture':(40,350),'Office Supplies':(3,40),'Apparel':(10,60)}
for cat, items in categories.items():
    for item in items:
        lo, hi = base_price[cat]
        products.append({
            'product_id': f"P{str(pid).zfill(4)}",
            'product_name': item,
            'category': cat,
            'unit_cost': round(np.random.uniform(lo, hi)*0.55, 2),
            'unit_price': round(np.random.uniform(lo, hi), 2)
        })
        pid += 1
products = pd.DataFrame(products)

# ---------- Orders / Order Items ----------
order_start = datetime(2023, 1, 1)
order_end = datetime(2024, 12, 31)
N_ORDERS = 6000

# give customers different "activity weights" so churn/retention patterns emerge naturally
activity_weight = np.random.exponential(scale=1.0, size=N_CUSTOMERS)
activity_weight = activity_weight / activity_weight.sum()

order_rows = []
item_rows = []
order_id_counter = 1

for i in range(N_ORDERS):
    cust_idx = np.random.choice(N_CUSTOMERS, p=activity_weight)
    cust_id = customer_ids[cust_idx]
    signup = signup_dates[cust_idx]
    earliest = max(order_start, signup)
    if earliest >= order_end:
        continue
    days_range = (order_end - earliest).days
    order_date = earliest + timedelta(days=np.random.randint(0, max(days_range,1)))

    # seasonal boost: Nov-Dec more orders (handled naturally via random draws + extra pass below)
    order_id = f"ORD{str(order_id_counter).zfill(6)}"
    order_id_counter += 1

    n_items = np.random.choice([1,2,3,4], p=[0.45,0.3,0.17,0.08])
    chosen_products = products.sample(n_items, replace=False)
    order_total = 0
    for _, prod in chosen_products.iterrows():
        qty = np.random.choice([1,2,3], p=[0.7,0.2,0.1])
        discount = np.random.choice([0,0.05,0.1,0.15,0.2], p=[0.5,0.2,0.15,0.1,0.05])
        line_total = round(prod['unit_price']*qty*(1-discount), 2)
        order_total += line_total
        item_rows.append({
            'order_id': order_id,
            'product_id': prod['product_id'],
            'quantity': qty,
            'discount': discount,
            'line_total': line_total
        })

    order_rows.append({
        'order_id': order_id,
        'customer_id': cust_id,
        'order_date': order_date,
        'order_total': round(order_total,2)
    })

orders = pd.DataFrame(order_rows)
order_items = pd.DataFrame(item_rows)

# Add a seasonal holiday boost by duplicating some Nov/Dec 2023 & 2024 orders with new ids/dates
holiday_boost_rows = []
holiday_item_rows = []
for year in [2023, 2024]:
    boost_orders = orders.sample(int(N_ORDERS*0.05))
    for _, row in boost_orders.iterrows():
        new_id = f"ORD{str(order_id_counter).zfill(6)}"
        order_id_counter += 1
        new_date = datetime(year, 12, np.random.randint(1,28))
        holiday_boost_rows.append({
            'order_id': new_id, 'customer_id': row['customer_id'],
            'order_date': new_date, 'order_total': row['order_total']
        })
        items_for_order = order_items[order_items['order_id']==row['order_id']]
        for _, it in items_for_order.iterrows():
            holiday_item_rows.append({**it.to_dict(), 'order_id': new_id})

orders = pd.concat([orders, pd.DataFrame(holiday_boost_rows)], ignore_index=True)
order_items = pd.concat([order_items, pd.DataFrame(holiday_item_rows)], ignore_index=True)
orders = orders.sort_values('order_date').reset_index(drop=True)

# Save
customers.to_csv('data/customers.csv', index=False)
products.to_csv('data/products.csv', index=False)
orders.to_csv('data/orders.csv', index=False)
order_items.to_csv('data/order_items.csv', index=False)

print("customers:", customers.shape)
print("products:", products.shape)
print("orders:", orders.shape)
print("order_items:", order_items.shape)
print(orders['order_date'].min(), orders['order_date'].max())
