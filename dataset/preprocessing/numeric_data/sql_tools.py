"""
sql_tools.py - Akses aman ke database SQLite Olist (data/olist.db) untuk SQL Agent.

Lapisan keamanan saat menjalankan SQL dari LLM:
  1. validate_query  : hanya SATU perintah SELECT / WITH; kata kunci tulis & bypass (main., temp., sqlite_,
                       ATTACH, PRAGMA, ...) ditolak.
  2. koneksi read-only (mode=ro): database tidak bisa diubah sama sekali.
  3. scope per role  : untuk seller/customer dibuat TEMP VIEW dengan nama yang sama dengan tabel asli,
                       sudah difilter ke pengguna itu. SQLite mencari di schema temp lebih dulu, jadi SQL biasa
                       (FROM order_items ...) hanya melihat baris milik pengguna itu.
  4. authorizer      : untuk seller/customer, membaca tabel main secara langsung (di luar view) ditolak,
                       kecuali tabel umum (products, br_states). Ini menutup celah seperti "main"."order_items".
  5. batas waktu, memori & baris: query terlalu lama dihentikan; satu nilai maksimal 10 MB;
                       hasil dipotong di MAX_ROWS (ditandai truncated).
Untuk seller, orders.items_value/freight_total/n_items/order_total dihitung ulang dari barang seller itu saja,
dan order_payments.payment_value dibagi sesuai porsi seller itu dalam order (order bisa berisi barang seller lain).

Fungsi utama:
  get_connection()                 -> koneksi read-only (untuk query internal yang sudah aman / berparameter)
  get_schema()                     -> deskripsi skema untuk prompt LLM
  make_scope(role, user_id)        -> Scope tervalidasi
  run_query(sql, scope)            -> DataFrame (attrs["truncated"] = True kalau dipotong)
  get_monthly_revenue(role, uid)   -> DataFrame tren pendapatan bulanan (untuk grafik / forecasting)
"""
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.getenv("OLIST_DB_PATH", ROOT / "olist.db"))

MAX_ROWS = 50            # baris maksimum yang dikembalikan ke LLM
QUERY_TIMEOUT_S = 15     # detik maksimum per query
MAX_VALUE_BYTES = 10_000_000   # ukuran maksimum satu nilai string/blob (batas memori)

ROLES = ("company", "seller", "customer")
ID_PATTERN = re.compile(r"[0-9a-f]{32}")     # seller_id / customer_unique_id Olist
PUBLIC_TABLES = {"products", "br_states"}    # data produk umum: boleh dibaca semua role

# Kata yang tidak boleh ada di SQL dari LLM (dicek sebagai kata utuh, di luar string literal)
FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|ANALYZE|"
    r"TRANSACTION|BEGIN|COMMIT|ROLLBACK|SAVEPOINT|RELEASE|LOAD_EXTENSION)\b"
    r"|\b(main|temp)\s*\.|\bsqlite_\w+",
    re.IGNORECASE,
)

# Penjelasan kolom yang tidak jelas dari namanya saja (masuk ke prompt LLM lewat get_schema)
COLUMN_NOTES = {
    "orders.is_delivered": "1 = status delivered AND delivery date recorded, else 0",
    "orders.is_late": "1 late / 0 on time (compared by date); NULL if not delivered",
    "orders.delivery_days": "days from purchase to delivery; NULL if not delivered",
    "orders.purchase_month": "'YYYY-MM' of order_purchase_timestamp",
    "orders.purchase_year": "year of order_purchase_timestamp",
    "orders.in_trend_period": "1 if purchase_month is between 2017-01 and 2018-08 (default analysis window)",
    "orders.items_value": "SUM(order_items.price) of the order",
    "orders.freight_total": "SUM(order_items.freight_value) of the order",
    "orders.n_items": "number of items in the order",
    "orders.order_total": "items_value + freight_total",
    "order_payments.payment_value": "NULL for zero-value vouchers (so AVG skips them)",
    "order_reviews.review_score": "1-5, exactly one review per order (first review kept)",
    "products.product_category_name_english": "'uncategorized' when the category is missing",
    "customers.customer_state": "2-letter code, join br_states.state_code for the full name",
    "sellers.seller_state": "2-letter code, join br_states.state_code for the full name",
    "sellers.seller_city": "cleaned: lowercase without accents, e.g. 'sao paulo'",
    "customers.customer_city": "lowercase without accents, e.g. 'sao paulo'",
}

# Statistik kategori se-platform (data produk umum, tanpa seller_id/customer) -> boleh untuk semua role.
# Semua tabel ditulis dengan prefix main. supaya view ini tidak ikut terfilter oleh scope.
CATEGORY_STATS_VIEW = """
CREATE TEMP VIEW category_stats AS
WITH items AS (
    SELECT oi.order_id, p.product_category_name_english AS category, oi.price
    FROM main.order_items oi
    JOIN main.products p ON p.product_id = oi.product_id
    JOIN main.orders o   ON o.order_id = oi.order_id
    WHERE o.is_delivered = 1 AND o.in_trend_period = 1
),
sales AS (
    SELECT category, COUNT(*) AS items_sold, COUNT(DISTINCT order_id) AS orders,
           ROUND(AVG(price), 2) AS avg_price
    FROM items GROUP BY category
),
reviews AS (
    SELECT x.category, COUNT(*) AS review_count, ROUND(AVG(x.review_score), 3) AS avg_review_score
    FROM (SELECT DISTINCT i.order_id, i.category, r.review_score
          FROM items i JOIN main.order_reviews r ON r.order_id = i.order_id) x
    GROUP BY x.category
)
SELECT s.category, s.items_sold, s.orders, s.avg_price,
       COALESCE(r.review_count, 0) AS review_count, r.avg_review_score
FROM sales s LEFT JOIN reviews r ON r.category = s.category
"""
CATEGORY_STATS_NOTE = (
    "VIEW category_stats (platform-wide, delivered orders in 2017-01..2018-08, one review per order): "
    "category TEXT, items_sold INTEGER, orders INTEGER, avg_price REAL, review_count INTEGER, "
    "avg_review_score REAL"
)


@dataclass(frozen=True)
class Scope:
    role: str = "company"
    user_id: str | None = None

    @property
    def is_restricted(self):
        return self.role in ("seller", "customer")


# =============================================================================
# KONEKSI & SKEMA
# =============================================================================
def get_connection():
    """Koneksi read-only ke olist.db. Error yang jelas kalau database belum dibuat."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found. Run: python data_preprocessing/numeric_data/clean_olist.py")
    return sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)


@lru_cache(maxsize=1)
def get_schema():
    """Teks skema (tabel, kolom, tipe, catatan) untuk prompt LLM. Dibaca sekali lalu di-cache."""
    conn = get_connection()
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        lines = []
        for table in tables:
            cols = []
            for _, name, ctype, *_ in conn.execute(f'PRAGMA table_info("{table}")'):
                note = COLUMN_NOTES.get(f"{table}.{name}")
                cols.append(f"  {name} {ctype}" + (f"  -- {note}" if note else ""))
            lines.append(f"TABLE {table}:\n" + "\n".join(cols))
        lines.append(CATEGORY_STATS_NOTE)
        return "\n\n".join(lines)
    finally:
        conn.close()


# =============================================================================
# SCOPE PER ROLE
# =============================================================================
def make_scope(role="company", user_id=None):
    """Validasi role + id. Seller/customer WAJIB punya id yang valid dan ada di database."""
    role = (role or "company").strip().lower()
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}; use one of {ROLES}.")
    if role == "company":
        return Scope("company")
    user_id = (user_id or "").strip().lower()
    if not user_id:
        raise ValueError(f"A {role} id is required for the {role} role.")
    if not ID_PATTERN.fullmatch(user_id):      # dicek dulu sebelum id masuk ke teks SQL view
        raise ValueError(f"Invalid {role} id: {user_id!r}")
    conn = get_connection()
    try:
        if role == "seller":
            found = conn.execute(
                "SELECT seller_id FROM sellers WHERE seller_id = ? LIMIT 1",
                (user_id,),
            ).fetchone()
        else:
            found = conn.execute(
                "SELECT customer_unique_id FROM customers "
                "WHERE customer_unique_id = ? OR customer_id = ? LIMIT 1",
                (user_id, user_id),
            ).fetchone()
    finally:
        conn.close()
    if not found:
        raise ValueError(f"Unknown {role} id: {user_id}")
    return Scope(role, found[0])


def _scope_views(scope):
    """CREATE TEMP VIEW untuk scope. Id sudah lolos ID_PATTERN (hanya hex), jadi aman ditulis ke teks SQL."""
    uid = scope.user_id
    if scope.role == "seller":
        own_orders = f"SELECT order_id FROM main.order_items WHERE seller_id = '{uid}'"
        # Nilai uang per order dihitung ulang dari barang seller ini saja (order bisa berisi barang seller lain)
        own_totals = ("SELECT order_id, ROUND(SUM(price), 2) AS items_value, "
                      "ROUND(SUM(freight_value), 2) AS freight_total, COUNT(*) AS n_items, "
                      "ROUND(SUM(price) + SUM(freight_value), 2) AS order_total "
                      f"FROM main.order_items WHERE seller_id = '{uid}' GROUP BY order_id")
        order_cols = ", ".join(f"o.{c}" for c in _order_columns() if c not in ORDER_MONEY_COLUMNS)
        return [
            f"CREATE TEMP VIEW sellers AS SELECT * FROM main.sellers WHERE seller_id = '{uid}'",
            f"CREATE TEMP VIEW order_items AS SELECT * FROM main.order_items WHERE seller_id = '{uid}'",
            f"CREATE TEMP VIEW orders AS SELECT {order_cols}, t.items_value, t.freight_total, t.n_items, "
            f"t.order_total FROM main.orders o JOIN ({own_totals}) t ON t.order_id = o.order_id",
            # Pembayaran tidak bisa dipisah per seller: payment_value dibagi sesuai porsi barang seller ini
            "CREATE TEMP VIEW order_payments AS SELECT p.order_id, p.payment_sequential, p.payment_type, "
            "p.payment_installments, ROUND(p.payment_value * t.order_total / o.order_total, 2) AS payment_value "
            f"FROM main.order_payments p JOIN ({own_totals}) t ON t.order_id = p.order_id "
            "JOIN main.orders o ON o.order_id = p.order_id",
            f"CREATE TEMP VIEW order_reviews AS SELECT * FROM main.order_reviews WHERE order_id IN ({own_orders})",
            "CREATE TEMP VIEW customers AS SELECT * FROM main.customers WHERE customer_id IN "
            f"(SELECT customer_id FROM main.orders WHERE order_id IN ({own_orders}))",
        ]
    if scope.role == "customer":
        own_ids = f"SELECT customer_id FROM main.customers WHERE customer_unique_id = '{uid}'"
        own_orders = f"SELECT order_id FROM main.orders WHERE customer_id IN ({own_ids})"
        return [
            f"CREATE TEMP VIEW customers AS SELECT * FROM main.customers WHERE customer_unique_id = '{uid}'",
            f"CREATE TEMP VIEW orders AS SELECT * FROM main.orders WHERE customer_id IN ({own_ids})",
            f"CREATE TEMP VIEW order_items AS SELECT * FROM main.order_items WHERE order_id IN ({own_orders})",
            f"CREATE TEMP VIEW order_payments AS SELECT * FROM main.order_payments WHERE order_id IN ({own_orders})",
            f"CREATE TEMP VIEW order_reviews AS SELECT * FROM main.order_reviews WHERE order_id IN ({own_orders})",
            "CREATE TEMP VIEW sellers AS SELECT * FROM main.sellers WHERE seller_id IN "
            f"(SELECT seller_id FROM main.order_items WHERE order_id IN ({own_orders}))",
        ]
    return []


ORDER_MONEY_COLUMNS = ("items_value", "freight_total", "n_items", "order_total")


@lru_cache(maxsize=1)
def _order_columns():
    """Nama kolom tabel orders (untuk membangun view seller)."""
    conn = get_connection()
    try:
        return tuple(row[1] for row in conn.execute("PRAGMA table_info(orders)"))
    finally:
        conn.close()


def _authorizer(scope):
    """Hanya izinkan membaca. Untuk seller/customer: tabel main hanya boleh dibaca lewat view scope."""
    def check(action, arg1, arg2, db_name, source):
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            if arg1 and arg1.lower().startswith("sqlite_"):
                return sqlite3.SQLITE_DENY
            if scope.is_restricted and db_name == "main" and source is None and arg1 not in PUBLIC_TABLES:
                return sqlite3.SQLITE_DENY       # baca tabel asli langsung = mencoba melewati scope
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY               # PRAGMA, ATTACH, tulis, buat objek, dst.
    return check


# =============================================================================
# VALIDASI & EKSEKUSI
# =============================================================================
def _strip_literals(sql):
    """Ganti isi string '...' dan komentar dengan spasi supaya cek kata kunci tidak salah tangkap."""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def validate_query(sql):
    """Kembalikan SQL yang sudah dibersihkan, atau raise ValueError kalau tidak boleh dijalankan."""
    if not sql or not sql.strip():
        raise ValueError("Empty query.")
    sql = sql.strip().rstrip(";").strip()
    bare = _strip_literals(sql)
    if ";" in bare:
        raise ValueError("Only ONE statement is allowed.")
    if not re.match(r"^\s*(SELECT|WITH)\b", bare, re.IGNORECASE):
        raise ValueError("Only read-only SELECT (or WITH ... SELECT) queries are allowed.")
    bad = FORBIDDEN.search(bare)
    if bad:
        raise ValueError(f"Forbidden keyword or object in query: {bad.group(0)!r}")
    return sql


def run_query(sql, scope=None, max_rows=MAX_ROWS, timeout=QUERY_TIMEOUT_S):
    """Jalankan SQL dari LLM dengan semua lapisan keamanan. Raise error kalau ditolak/gagal."""
    sql = validate_query(sql)
    scope = scope or Scope("company")
    conn = get_connection()
    try:
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)   # cegah zeroblob/teks raksasa di memori
        for view_sql in _scope_views(scope) + [CATEGORY_STATS_VIEW]:
            conn.execute(view_sql)
        conn.set_authorizer(_authorizer(scope))
        deadline = time.monotonic() + timeout
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cur = conn.execute(sql)
            rows = cur.fetchmany(max_rows + 1)      # 1 baris ekstra = tanda masih ada baris lain
        except sqlite3.OperationalError as error:
            if "interrupted" in str(error):
                raise TimeoutError(f"Query took longer than {timeout} seconds; simplify it.") from None
            raise
        columns = [d[0] for d in cur.description]
    finally:
        conn.close()
    data = pd.DataFrame(rows[:max_rows], columns=columns)
    data.attrs["truncated"] = len(rows) > max_rows
    return data


# =============================================================================
# QUERY TETAP (tanpa LLM) untuk grafik & forecasting
# =============================================================================
MONTHLY_REVENUE_SQL = """
SELECT o.purchase_month AS month,
       ROUND(SUM(oi.price), 2)      AS revenue,
       COUNT(DISTINCT o.order_id)   AS orders
FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.is_delivered = 1 AND o.in_trend_period = 1
GROUP BY o.purchase_month
ORDER BY o.purchase_month
"""


def get_monthly_revenue(role="company", user_id=None):
    """Pendapatan bulanan (order delivered, 2017-01..2018-08), sudah dibatasi sesuai role."""
    return run_query(MONTHLY_REVENUE_SQL, scope=make_scope(role, user_id), max_rows=1000)
