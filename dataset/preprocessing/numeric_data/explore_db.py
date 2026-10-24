import sqlite3
import pandas as pd

conn = sqlite3.connect("data/olist.db")

def run(title, query):
    print(f"\n=== {title} ===")
    print(pd.read_sql_query(query, conn).to_string(index=False))

run("1. Daftar tabel", """
SELECT name FROM sqlite_master WHERE type = 'table'
""")

run("2. Jumlah order per status", """
SELECT order_status, COUNT(*) AS jumlah
FROM orders
GROUP BY order_status
ORDER BY jumlah DESC
""")

run("3. Rata-rata skor review", """
SELECT ROUND(AVG(review_score), 2) AS rata_rata_skor
FROM order_reviews
""")

run("4. Top 5 kategori produk berdasarkan pendapatan (order delivered)", """
SELECT p.product_category_name_english AS kategori,
       ROUND(SUM(oi.price), 2) AS pendapatan
FROM order_items oi
JOIN orders o ON o.order_id = oi.order_id
JOIN products p ON p.product_id = oi.product_id
WHERE o.is_delivered = 1
GROUP BY kategori
ORDER BY pendapatan DESC
LIMIT 5
""")

conn.close()