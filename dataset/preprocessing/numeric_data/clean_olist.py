"""
clean_olist.py - Cleaning dataset Olist (CSV) menjadi database SQLite untuk SQL Agent.

Prinsip: file CSV mentah tidak diubah; hasil bersih ditulis ke file olist.db yang baru
(file lama dihapus dan dibuat ulang setiap kali script dijalankan).

Argumen command line:
    --data-dir     folder yang berisi raw_data/*.csv (default ./dataset)
    --out-dir      folder tujuan olist.db (default ./dataset)
    --review-pick  ulasan yang dipakai kalau satu order punya lebih dari satu ulasan:
                   "first" (default) atau "latest"

Aturan cleaning:
    - Order tidak dihapus, hanya ditandai. Revenue = SUM(order_items.price) dari order delivered.
    - Periode analisis 2017-01 s.d. 2018-08 (kolom purchase_month, purchase_year, in_trend_period).
    - Ulasan ganda: ulasan PERTAMA per order yang dipakai; teks ulasan tidak disimpan.
    - Produk tanpa kategori diberi label 'uncategorized'; dua kategori yang belum punya terjemahan ditambahkan.
    - Pembayaran: tipe not_defined dibuang, cicilan 0 -> 1, voucher bernilai 0 -> NULL.
    - Nama kota seller dibersihkan (clean_city); data customer tidak diubah.
    - is_late dibandingkan per tanggal, karena estimasi pengiriman selalu tercatat jam 00:00.

Kolom tambahan di tabel orders:
    is_delivered, purchase_month, purchase_year, in_trend_period, is_late, delivery_days,
    items_value, freight_total, n_items, order_total.

Tabel di olist.db:
    customers, sellers, br_states, products, orders, order_items, order_payments, order_reviews.
"""
import argparse
import os
import re
import sqlite3
import unicodedata

import numpy as np
import pandas as pd

# Periode analisis (tren); data sebelum 2017-01 dan sesudah 2018-08 tidak lengkap
TREND_START, TREND_END = "2017-01", "2018-08"

# Label untuk produk tanpa kategori
UNCATEGORIZED = "uncategorized"

# Terjemahan untuk dua kategori yang tidak ada di file product_category_name_translation.csv
EXTRA_TRANSLATIONS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_and_food_preparers",
}

# Kode negara bagian Brasil -> (nama negara bagian, wilayah); disimpan sebagai tabel br_states
BR_STATES = {
    "AC": ("Acre", "North"), "AL": ("Alagoas", "Northeast"), "AP": ("Amapá", "North"),
    "AM": ("Amazonas", "North"), "BA": ("Bahia", "Northeast"), "CE": ("Ceará", "Northeast"),
    "DF": ("Distrito Federal", "Central-West"), "ES": ("Espírito Santo", "Southeast"),
    "GO": ("Goiás", "Central-West"), "MA": ("Maranhão", "Northeast"), "MT": ("Mato Grosso", "Central-West"),
    "MS": ("Mato Grosso do Sul", "Central-West"), "MG": ("Minas Gerais", "Southeast"),
    "PA": ("Pará", "North"), "PB": ("Paraíba", "Northeast"), "PR": ("Paraná", "South"),
    "PE": ("Pernambuco", "Northeast"), "PI": ("Piauí", "Northeast"), "RJ": ("Rio de Janeiro", "Southeast"),
    "RN": ("Rio Grande do Norte", "Northeast"), "RS": ("Rio Grande do Sul", "South"),
    "RO": ("Rondônia", "North"), "RR": ("Roraima", "North"), "SC": ("Santa Catarina", "South"),
    "SP": ("São Paulo", "Southeast"), "SE": ("Sergipe", "Northeast"), "TO": ("Tocantins", "North"),
}


# Salah ketik nama kota seller yang diperbaiki manual. Sengaja tidak memakai pencocokan fuzzy otomatis,
# karena ada kota asli yang namanya mirip kota lain (mis. 'sao jose do rio pardo' dan 'itapeva').
CITY_FIXES = {
    "sp": "sao paulo", "sao paulo sp": "sao paulo", "sao pauo": "sao paulo", "sao paluo": "sao paulo",
    "sao paulop": "sao paulo", "sbc": "sao bernardo do campo", "sao bernardo do capo": "sao bernardo do campo",
    "ao bernardo do campo": "sao bernardo do campo", "belo horizont": "belo horizonte", "cascavael": "cascavel",
    "sao jose dos pinhas": "sao jose dos pinhais", "balenario camboriu": "balneario camboriu",
    "floranopolis": "florianopolis", "mogi das cruses": "mogi das cruzes",
    "s jose do rio preto": "sao jose do rio preto", "sao jose do rio pret": "sao jose do rio preto",
    "scao jose do rio pardo": "sao jose do rio pardo", "portoferreira": "porto ferreira",
    "sando andre": "santo andre", "garulhos": "guarulhos", "ribeirao pretp": "ribeirao preto",
    "santa barbara d oeste": "santa barbara d'oeste", "sao miguel d'oeste": "sao miguel do oeste",
    "robeirao preto": "ribeirao preto", "riberao preto": "ribeirao preto", "tabao da serra": "taboao da serra",
}


def clean_city(city):
    """Seragamkan nama kota: huruf kecil tanpa aksen, buang akhiran seperti '/ sp', '- sp', ', brasil', '(...)',
    lalu perbaiki salah ketik lewat CITY_FIXES.
    Mengembalikan None kalau isinya bukan nama kota (mis. angka atau email)."""
    if pd.isna(city):
        return None
    city = unicodedata.normalize("NFKD", str(city)).encode("ascii", "ignore").decode().lower()
    city = city.replace("´", "'").replace("`", "'")
    city = re.split(r"\s*[/\\,(]\s*|\s+-\s+|-(?=[a-z]{2}$)", city)[0]
    city = re.sub(r"\s+", " ", city).strip()
    city = CITY_FIXES.get(city, city)
    return city if re.fullmatch(r"[a-z][a-z' ]+", city) else None


def clean_seller_cities(sellers):
    """Bersihkan kolom seller_city dengan clean_city.
    Kalau hasilnya None, isi dengan kota yang paling sering dipakai seller lain dengan zip prefix yang sama;
    kalau tetap tidak ada, isi 'unknown'."""
    sellers["seller_city"] = sellers["seller_city"].map(clean_city)
    by_zip = (sellers.dropna(subset=["seller_city"])
              .groupby("seller_zip_code_prefix")["seller_city"].agg(lambda c: c.value_counts().index[0]))
    missing = sellers["seller_city"].isna()
    sellers.loc[missing, "seller_city"] = sellers.loc[missing, "seller_zip_code_prefix"].map(by_zip)
    sellers["seller_city"] = sellers["seller_city"].fillna("unknown")
    return sellers


def read(data_dir, name, **kw):
    """Baca satu file CSV dari data_dir (encoding utf-8-sig supaya BOM di awal file ikut terbuang)."""
    return pd.read_csv(os.path.join(data_dir, name), encoding="utf-8-sig", **kw)


def fmt_dates(df, cols):
    """Ubah kolom tanggal menjadi TEXT 'YYYY-MM-DD HH:MM:SS' (NULL kalau kosong), format yang dipakai di SQLite."""
    for c in cols:
        df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def main():
    """Baca semua CSV, bersihkan, tulis ke olist.db, buat index, lalu cetak ringkasan validasi."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./dataset")
    parser.add_argument("--out-dir", default="./dataset")
    parser.add_argument("--review-pick", choices=["first", "latest"], default="first",
                        help="ulasan mana yang dipakai kalau 1 order punya >1 ulasan")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    d = args.data_dir

    # ------------------------------------------------------------------ CUSTOMERS (tidak diubah)
    customers = read(d, "raw_data/olist_customers_dataset.csv", dtype={"customer_zip_code_prefix": str})

    # ------------------------------------------------------------------ SELLERS (nama kota dibersihkan)
    sellers = read(d, "raw_data/olist_sellers_dataset.csv", dtype={"seller_zip_code_prefix": str})
    raw_cities = sellers["seller_city"].nunique()
    sellers = clean_seller_cities(sellers)

    # ------------------------------------------------------------------ BR_STATES
    br_states = pd.DataFrame([(k, v[0], v[1]) for k, v in BR_STATES.items()],
                             columns=["state_code", "state_name", "region"])

    # ------------------------------------------------------------------ PRODUCTS (kategori + terjemahan; dimensi & berat dibuang)
    products = read(d, "raw_data/olist_products_dataset.csv")
    translation = read(d, "raw_data/product_category_name_translation.csv")
    translation.columns = [c.strip() for c in translation.columns]
    tmap = dict(zip(translation.product_category_name, translation.product_category_name_english))
    tmap.update(EXTRA_TRANSLATIONS)
    products["product_category_name"] = products["product_category_name"].fillna(UNCATEGORIZED)
    products["product_category_name_english"] = (
        products["product_category_name"].map(tmap).fillna(UNCATEGORIZED))
    products = products[["product_id", "product_category_name", "product_category_name_english"]]

    # ------------------------------------------------------------------ ORDER ITEMS
    items = read(d, "raw_data/olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
    items = fmt_dates(items, ["shipping_limit_date"])

    # ------------------------------------------------------------------ PAYMENTS
    payments = read(d, "raw_data/olist_order_payments_dataset.csv")
    payments = payments[payments["payment_type"] != "not_defined"].copy()           # buang 3 baris (order canceled)
    payments.loc[payments["payment_installments"] == 0, "payment_installments"] = 1  # cicilan 0 -> 1 (2 baris)
    # voucher bernilai 0 -> NULL: SUM tidak berubah, AVG otomatis melewatinya
    payments["payment_value"] = payments["payment_value"].astype(float)
    payments.loc[(payments["payment_type"] == "voucher") & (payments["payment_value"] == 0),
                 "payment_value"] = np.nan

    # ------------------------------------------------------------------ REVIEWS (1 ulasan per order, tanpa teks)
    reviews = read(d, "raw_data/olist_order_reviews_dataset.csv",
                   parse_dates=["review_creation_date", "review_answer_timestamp"])
    reviews = reviews[["review_id", "order_id", "review_score",
                       "review_creation_date", "review_answer_timestamp"]]           # tanpa teks
    if args.review_pick == "first":    # default: ulasan pertama = pendapat final pembeli
        reviews = reviews.sort_values(["order_id", "review_creation_date", "review_answer_timestamp"]
                                      ).drop_duplicates("order_id", keep="first")
    else:                              # alternatif: ulasan dengan review_answer_timestamp terbaru
        reviews = reviews.sort_values(["order_id", "review_answer_timestamp"]).drop_duplicates("order_id", keep="last")
    reviews = fmt_dates(reviews, ["review_creation_date", "review_answer_timestamp"])

    # ------------------------------------------------------------------ ORDERS (+ kolom turunan)
    date_cols = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                 "order_delivered_customer_date", "order_estimated_delivery_date"]
    orders = read(d, "raw_data/olist_orders_dataset.csv", parse_dates=date_cols)

    # Order tidak dihapus, cukup ditandai. Delivered = status delivered DAN tanggal sampai tercatat.
    orders["is_delivered"] = ((orders["order_status"] == "delivered")
                              & orders["order_delivered_customer_date"].notna()).astype(int)

    # Kolom periode dibuat saat cleaning, supaya agent tidak perlu memotong timestamp sendiri
    orders["purchase_month"] = orders["order_purchase_timestamp"].dt.strftime("%Y-%m")
    orders["purchase_year"] = orders["order_purchase_timestamp"].dt.year.astype(int)
    orders["in_trend_period"] = orders["purchase_month"].between(TREND_START, TREND_END).astype(int)

    # Pengiriman: tanggal estimasi selalu jam 00:00, jadi bandingkan TANGGAL-nya saja.
    # is_late dan delivery_days hanya diisi untuk order delivered; order lain NULL.
    delivered = orders["is_delivered"] == 1
    orders["is_late"] = np.where(
        delivered,
        (orders["order_delivered_customer_date"].dt.normalize()
         > orders["order_estimated_delivery_date"].dt.normalize()).astype(float),
        np.nan)
    orders["delivery_days"] = np.where(
        delivered,
        ((orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"])
         .dt.total_seconds() / 86400).round(1),
        np.nan)

    # Ringkasan nilai per order dari order_items (payment_value tidak dipakai untuk revenue)
    per_order = items.groupby("order_id").agg(items_value=("price", "sum"),
                                              freight_total=("freight_value", "sum"),
                                              n_items=("order_item_id", "count")).reset_index()
    orders = orders.merge(per_order, on="order_id", how="left")
    orders[["items_value", "freight_total", "n_items"]] = orders[["items_value", "freight_total", "n_items"]].fillna(0)
    orders["n_items"] = orders["n_items"].astype(int)                       # jumlah barang = bilangan bulat
    orders["items_value"] = orders["items_value"].round(2)
    orders["freight_total"] = orders["freight_total"].round(2)
    orders["order_total"] = (orders["items_value"] + orders["freight_total"]).round(2)
    orders = fmt_dates(orders, date_cols)

    # ------------------------------------------------------------------ TULIS KE SQLITE
    db_path = os.path.join(args.out_dir, "olist.db")
    if os.path.exists(db_path):          # database selalu dibuat ulang dari awal
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    tables = {"customers": customers, "sellers": sellers, "br_states": br_states, "products": products,
              "orders": orders, "order_items": items, "order_payments": payments, "order_reviews": reviews}
    for name, df in tables.items():
        df.to_sql(name, conn, index=False)

    # Index: mempercepat JOIN / filter dan memastikan kolom kunci unik
    indexes = [
        "CREATE UNIQUE INDEX ix_customers_pk  ON customers(customer_id)",
        "CREATE INDEX        ix_customers_uid ON customers(customer_unique_id)",
        "CREATE UNIQUE INDEX ix_sellers_pk    ON sellers(seller_id)",
        "CREATE UNIQUE INDEX ix_products_pk   ON products(product_id)",
        "CREATE UNIQUE INDEX ix_orders_pk     ON orders(order_id)",
        "CREATE INDEX        ix_orders_cust   ON orders(customer_id)",
        "CREATE INDEX        ix_orders_month  ON orders(purchase_month)",
        "CREATE UNIQUE INDEX ix_items_pk      ON order_items(order_id, order_item_id)",
        "CREATE INDEX        ix_items_seller  ON order_items(seller_id)",
        "CREATE INDEX        ix_items_product ON order_items(product_id)",
        "CREATE UNIQUE INDEX ix_pay_pk        ON order_payments(order_id, payment_sequential)",
        "CREATE UNIQUE INDEX ix_reviews_pk    ON order_reviews(review_id, order_id)",
        "CREATE UNIQUE INDEX ix_reviews_order ON order_reviews(order_id)",              # 1 ulasan per order
    ]
    for sql in indexes:
        conn.execute(sql)
    conn.commit()

    # ------------------------------------------------------------------ VALIDASI SINGKAT
    q = lambda sql: conn.execute(sql).fetchone()[0]       # jalankan query, ambil satu nilai
    print("database                  :", db_path, f"({os.path.getsize(db_path) / 1e6:.0f} MB)")
    for name in tables:
        print(f"  {name:<24}:", q(f"SELECT COUNT(*) FROM {name}"))
    print("orders delivered          :", q("SELECT SUM(is_delivered) FROM orders"))
    print("revenue delivered (R$)    :", q("SELECT ROUND(SUM(oi.price),2) FROM order_items oi "
                                           "JOIN orders o USING(order_id) WHERE o.is_delivered=1"))
    print("on-time rate              :", q("SELECT ROUND(1-AVG(is_late),4) FROM orders WHERE is_delivered=1"))
    print("avg review_score          :", q("SELECT ROUND(AVG(review_score),3) FROM order_reviews"))
    print("produk uncategorized      :", q(f"SELECT COUNT(*) FROM products "
                                           f"WHERE product_category_name_english='{UNCATEGORIZED}'"))
    print("bulan dalam tren          :", q("SELECT COUNT(DISTINCT purchase_month) FROM orders WHERE in_trend_period=1"))
    print("kota seller (mentah->bersih):", raw_cities, "->", q("SELECT COUNT(DISTINCT seller_city) FROM sellers"))
    print("review pick               :", args.review_pick)
    conn.close()


if __name__ == "__main__":
    main()
