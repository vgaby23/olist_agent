"""
Cleaning dataset Olist -> SQLite (untuk SQL Agent) + dokumen review (untuk RAG Agent / Qdrant)

Cara pakai:
    python clean_olist.py --data-dir ./OlistDataset --out-dir ./output

Hasil:
    output/olist.db        -> database SQLite bersih
    output/rag_docs.csv    -> dokumen review siap di-embed ke Qdrant
"""
import argparse
import os
import re
import sqlite3
import unicodedata

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="/mnt/project")
parser.add_argument("--out-dir", default="/mnt/user-data/outputs")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)


def read(name, **kw):
    return pd.read_csv(os.path.join(args.data_dir, name), **kw)


def iso(df, cols):
    """Datetime -> string ISO supaya bisa dipakai strftime() di SQLite."""
    for c in cols:
        df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


# ------------------------------------------------------------------ LOAD
customers = read("olist_customers_dataset.csv")
orders = read("olist_orders_dataset.csv")
items = read("olist_order_items_dataset.csv")
payments = read("olist_order_payments_dataset.csv")
reviews = read("olist_order_reviews_dataset.csv")
products = read("olist_products_dataset.csv")
sellers = read("olist_sellers_dataset.csv")
trans = read("product_category_name_translation.csv", encoding="utf-8-sig")
print("Loaded OK")

# ------------------------------------------------------------------ TRANSLATION + PRODUCTS
# 2 kategori yang tidak ada di file terjemahan
trans = pd.concat(
    [trans, pd.DataFrame({
        "product_category_name": ["portateis_cozinha_e_preparadores_de_alimentos", "pc_gamer"],
        "product_category_name_english": ["portable_kitchen_and_food_preparers", "pc_gamer"],
    })],
    ignore_index=True,
)
products = products.merge(trans, on="product_category_name", how="left")
products["product_category_name"] = products["product_category_name"].fillna("unknown")
products["product_category_name_english"] = products["product_category_name_english"].fillna("unknown")
# Sesuai PRD: dimensi, berat, panjang nama/deskripsi tidak dipakai
products = products[["product_id", "product_category_name", "product_category_name_english", "product_photos_qty"]]

# ------------------------------------------------------------------ SELLERS
def clean_city(x):
    x = str(x).lower().strip().replace("´", "'").replace("`", "'")
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode()
    if re.fullmatch(r"[\d\s\-]+", x):          # isinya kode pos
        return "unknown"
    x = re.split(r"\s*[/,\\(]\s*", x)[0].strip()  # buang "/sp", ", brasil", "(porto seguro)"
    x = {"sbc": "sao bernardo do campo"}.get(x, x)
    return x or "unknown"


sellers["seller_city"] = sellers["seller_city"].map(clean_city)

# ------------------------------------------------------------------ ORDERS
dt_cols = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
           "order_delivered_customer_date", "order_estimated_delivery_date"]
for c in dt_cols:
    orders[c] = pd.to_datetime(orders[c])

# tanggal tidak logis -> NULL
orders.loc[orders["order_delivered_carrier_date"] < orders["order_purchase_timestamp"],
           "order_delivered_carrier_date"] = pd.NaT
orders.loc[orders["order_status"] != "delivered", "order_delivered_customer_date"] = pd.NaT

# kolom turunan
orders["is_delivered"] = ((orders["order_status"] == "delivered")
                          & orders["order_delivered_customer_date"].notna()).astype(int)
delivery = orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
orders["delivery_days"] = (delivery.dt.total_seconds() / 86400).round(1)
orders["is_late"] = np.where(
    orders["is_delivered"] == 1,
    (orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]).astype(float),
    np.nan,
)
orders["purchase_month"] = orders["order_purchase_timestamp"].dt.strftime("%Y-%m")
# bulan 2016-09, 2016-12, 2018-09, 2018-10 hampir kosong -> jangan dipakai untuk tren/forecast
orders["in_trend_period"] = orders["purchase_month"].between("2017-01", "2018-08").astype(int)

# total nilai order dari item (harga + ongkir)
tot = items.groupby("order_id").agg(items_value=("price", "sum"), freight_total=("freight_value", "sum"),
                                    n_items=("order_item_id", "count")).reset_index()
orders = orders.merge(tot, on="order_id", how="left")
orders[["items_value", "freight_total", "n_items"]] = orders[["items_value", "freight_total", "n_items"]].fillna(0)
orders["order_total"] = (orders["items_value"] + orders["freight_total"]).round(2)

# ------------------------------------------------------------------ ORDER ITEMS
items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"])

# ------------------------------------------------------------------ PAYMENTS
payments = payments[payments["payment_value"] > 0].copy()      # buang not_defined & nilai 0
payments["payment_installments"] = payments["payment_installments"].clip(lower=1)

# ------------------------------------------------------------------ REVIEWS
reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"])
reviews["review_answer_timestamp"] = pd.to_datetime(reviews["review_answer_timestamp"])
# 1 review per order (yang terbaru)
reviews = (reviews.sort_values("review_answer_timestamp")
           .drop_duplicates("order_id", keep="last")
           .reset_index(drop=True))

# teks bersih untuk RAG
def clean_text(s):
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


reviews["review_comment_title"] = reviews["review_comment_title"].map(clean_text)
reviews["review_comment_message"] = reviews["review_comment_message"].map(clean_text)

# ------------------------------------------------------------------ RAG DOCS
rag = reviews[reviews["review_comment_message"].str.len() > 3].copy()
pair = (items.groupby(["order_id", "product_id", "seller_id"], as_index=False)
        .agg(price=("price", "mean")))
rag = (rag.merge(pair, on="order_id")
       .merge(products[["product_id", "product_category_name_english"]], on="product_id", how="left")
       .merge(sellers[["seller_id", "seller_city", "seller_state"]], on="seller_id", how="left"))
rag = rag.rename(columns={"product_category_name_english": "product_category"})


def build_doc(r):
    parts = [f"Category: {r.product_category}", f"Seller location: {r.seller_city}, {r.seller_state}"]
    if r.review_comment_title:
        parts.append(f"Title: {r.review_comment_title}")
    parts.append(f"Review: {r.review_comment_message}")
    return " | ".join(parts)


rag["page_content"] = rag.apply(build_doc, axis=1)
rag = rag[["review_id", "order_id", "product_id", "seller_id", "product_category",
           "seller_city", "seller_state", "review_score", "price", "page_content"]]
rag.to_csv(os.path.join(args.out_dir, "rag_docs.csv"), index=False)

# Tabel review untuk SQLite: skor saja, tanpa teks (sesuai PRD)
reviews_sql = reviews[["review_id", "order_id", "review_score",
                       "review_creation_date", "review_answer_timestamp"]].copy()

# ------------------------------------------------------------------ WRITE SQLITE
orders = iso(orders, dt_cols)
items = iso(items, ["shipping_limit_date"])
reviews_sql = iso(reviews_sql, ["review_creation_date", "review_answer_timestamp"])

db_path = os.path.join(args.out_dir, "olist.db")
if os.path.exists(db_path):
    os.remove(db_path)
conn = sqlite3.connect(db_path)
for name, df in [("customers", customers), ("orders", orders), ("order_items", items),
                 ("order_payments", payments), ("order_reviews", reviews_sql),
                 ("products", products), ("sellers", sellers)]:
    df.to_sql(name, conn, index=False)

for stmt in [
    "CREATE UNIQUE INDEX ix_cust_id ON customers(customer_id)",
    "CREATE INDEX ix_cust_unique ON customers(customer_unique_id)",
    "CREATE UNIQUE INDEX ix_ord_id ON orders(order_id)",
    "CREATE INDEX ix_ord_cust ON orders(customer_id)",
    "CREATE INDEX ix_ord_month ON orders(purchase_month)",
    "CREATE INDEX ix_item_order ON order_items(order_id)",
    "CREATE INDEX ix_item_prod ON order_items(product_id)",
    "CREATE INDEX ix_item_sell ON order_items(seller_id)",
    "CREATE INDEX ix_pay_order ON order_payments(order_id)",
    "CREATE UNIQUE INDEX ix_rev_order ON order_reviews(order_id)",
    "CREATE UNIQUE INDEX ix_prod_id ON products(product_id)",
    "CREATE UNIQUE INDEX ix_sell_id ON sellers(seller_id)",
]:
    conn.execute(stmt)
conn.commit()

# ------------------------------------------------------------------ VALIDASI
print("\n=== VALIDASI ===")
q = lambda s: conn.execute(s).fetchone()[0]
for t in ["customers", "orders", "order_items", "order_payments", "order_reviews", "products", "sellers"]:
    print(f"{t:15s}", q(f"SELECT COUNT(*) FROM {t}"))
print("review duplikat per order :", q("SELECT COUNT(*) FROM (SELECT order_id FROM order_reviews GROUP BY order_id HAVING COUNT(*)>1)"))
print("payment not_defined       :", q("SELECT COUNT(*) FROM order_payments WHERE payment_type='not_defined'"))
print("produk kategori unknown   :", q("SELECT COUNT(*) FROM products WHERE product_category_name_english='unknown'"))
print("order delivered valid     :", q("SELECT SUM(is_delivered) FROM orders"))
print("order di periode tren     :", q("SELECT SUM(in_trend_period) FROM orders"))
print("seller city aneh          :", q("SELECT COUNT(*) FROM sellers WHERE seller_city GLOB '*[^a-z '' -]*'"))
print("dokumen RAG               :", len(rag))
print("avg review_score          :", round(q("SELECT AVG(review_score) FROM order_reviews"), 3))
conn.close()
print("\nSelesai ->", args.out_dir)
