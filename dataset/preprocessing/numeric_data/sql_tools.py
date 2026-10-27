"""
sql_tools.py - Akses aman ke database SQLite Olist (olist.db) untuk SQL Agent.

Lokasi database diatur oleh DB_PATH: default olist.db di folder Dataset, atau path dari env OLIST_DB_PATH.

Lapisan keamanan saat menjalankan SQL buatan LLM:
  1. validate_query  : hanya boleh SATU perintah SELECT / WITH. Kata kunci untuk menulis data atau mengakali
                       pembatasan (INSERT, DROP, ATTACH, PRAGMA, main., temp., sqlite_, ...) ditolak.
  2. Koneksi read-only (mode=ro): isi database tidak bisa diubah sama sekali.
  3. Scope per role  : untuk seller dibuat TEMP VIEW dengan nama yang sama dengan tabel asli, yang sudah
                       difilter ke seller itu. SQLite mencari di schema temp lebih dulu, jadi query biasa
                       (FROM order_items ...) hanya melihat baris milik seller itu.
  4. Authorizer      : untuk seller/customer, membaca tabel asli secara langsung (di luar view) ditolak,
                       kecuali tabel umum (products, br_states). Ini menutup celah seperti "main"."order_items".
  5. Batas waktu, memori, dan baris: query yang terlalu lama dihentikan, satu nilai maksimal 10 MB,
                       dan hasil dipotong di MAX_ROWS baris (ditandai attrs["truncated"]).

Data yang bisa dilihat tiap role:
  - company : semua tabel dan view, dengan angka persis.
  - seller  : hanya transaksi miliknya sendiri.
      * orders.items_value / freight_total / n_items / order_total dihitung ulang dari barang seller itu saja.
      * order_payments.payment_value dibagi sesuai porsi seller itu dalam order
        (satu order bisa berisi barang dari beberapa seller).
      * customer_id / customer_unique_id diganti pseudonim dan kode pos dibuang: jumlah pembeli dan
        repeat purchase tetap benar, tetapi pembeli asli tidak bisa dikenali.
      * category_stats memakai angka persis; product_stats dan seller_stats memakai versi customer
        (tanpa angka persis), supaya omzet seller lain tidak terlihat.
  - customer: pembeli ANONIM (tanpa id). Tabel transaksi tidak bisa dibaca sama sekali; yang terbuka hanya
              tabel umum (products, br_states) dan view agregat se-platform (category_stats, product_stats,
              seller_stats tanpa angka persis, plus state_delivery_stats).

Fungsi utama:
  get_connection()                 -> koneksi read-only (untuk query internal yang aman / berparameter)
  get_schema(role)                 -> teks skema database untuk prompt LLM
  make_scope(role, user_id)        -> Scope yang sudah divalidasi
  run_query(sql, scope)            -> DataFrame hasil query (attrs["truncated"] = True kalau dipotong)
  get_monthly_revenue(role, uid)   -> DataFrame pendapatan bulanan (untuk grafik / forecasting)
"""
import hashlib
import hmac
import os
import re
import secrets
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
ID_PATTERN = re.compile(r"[0-9a-f]{32}")     # format id Olist: 32 karakter hex
PUBLIC_TABLES = {"products", "br_states"}    # tabel umum yang boleh dibaca semua role

# Kata kunci yang tidak boleh ada di SQL dari LLM (dicek sebagai kata utuh, setelah isi string & komentar dibuang)
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

# Statistik per kategori se-platform (tanpa data seller/customer), boleh untuk semua role.
# Semua tabel ditulis dengan prefix main. supaya view ini membaca tabel asli, bukan TEMP VIEW milik scope seller.
CATEGORY_STATS_SELECT = """
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
CATEGORY_STATS_VIEW = "CREATE TEMP VIEW category_stats AS" + CATEGORY_STATS_SELECT
CATEGORY_STATS_NOTE = (
    "VIEW category_stats (platform-wide, delivered orders in 2017-01..2018-08, one review per order): "
    "category TEXT, items_sold INTEGER, orders INTEGER, avg_price REAL, review_count INTEGER, "
    "avg_review_score REAL"
)

# Statistik per produk se-platform (tanpa data seller/customer). Versi angka persis ini hanya untuk company;
# seller dan customer memakai CUSTOMER_PRODUCT_STATS_VIEW.
PRODUCT_STATS_SELECT = """
WITH items AS (
    SELECT oi.order_id, oi.product_id, p.product_category_name_english AS category, oi.price
    FROM main.order_items oi
    JOIN main.products p ON p.product_id = oi.product_id
    JOIN main.orders o   ON o.order_id = oi.order_id
    WHERE o.is_delivered = 1 AND o.in_trend_period = 1
),
sales AS (
    SELECT product_id, category, COUNT(*) AS items_sold, ROUND(AVG(price), 2) AS avg_price
    FROM items GROUP BY product_id, category
),
reviews AS (
    SELECT x.product_id, COUNT(*) AS review_count, ROUND(AVG(x.review_score), 3) AS avg_review_score
    FROM (SELECT DISTINCT i.order_id, i.product_id, r.review_score
          FROM items i JOIN main.order_reviews r ON r.order_id = i.order_id) x
    GROUP BY x.product_id
)
SELECT s.product_id, s.category, s.items_sold, s.avg_price,
       COALESCE(r.review_count, 0) AS review_count, r.avg_review_score
FROM sales s LEFT JOIN reviews r ON r.product_id = s.product_id
"""
PRODUCT_STATS_VIEW = "CREATE TEMP VIEW product_stats AS" + PRODUCT_STATS_SELECT
PRODUCT_STATS_NOTE = (
    "VIEW product_stats (platform-wide per product, delivered orders in 2017-01..2018-08, one review per order): "
    "product_id TEXT, category TEXT, items_sold INTEGER, avg_price REAL, review_count INTEGER, avg_review_score REAL"
)

# Lama & ketepatan pengiriman per state pembeli se-platform (agregat, tanpa id), boleh untuk semua role.
STATE_DELIVERY_VIEW = """
CREATE TEMP VIEW state_delivery_stats AS
SELECT c.customer_state AS state_code, s.state_name,
       COUNT(*) AS delivered_orders,
       ROUND(AVG(o.delivery_days), 1) AS avg_delivery_days,
       ROUND(100.0 * AVG(o.is_late), 1) AS late_pct
FROM main.orders o
JOIN main.customers c ON c.customer_id = o.customer_id
LEFT JOIN main.br_states s ON s.state_code = c.customer_state
WHERE o.is_delivered = 1 AND o.in_trend_period = 1
GROUP BY c.customer_state, s.state_name
"""
STATE_DELIVERY_NOTE = (
    "VIEW state_delivery_stats (platform-wide per buyer state, delivered orders in 2017-01..2018-08): "
    "state_code TEXT, state_name TEXT, delivered_orders INTEGER, avg_delivery_days REAL, late_pct REAL"
)

# Statistik per seller per kategori se-platform (untuk rekomendasi seller, mis. "5 seller review terbaik untuk
# make up"). Versi angka persis ini hanya untuk company; seller dan customer memakai CUSTOMER_SELLER_STATS_VIEW.
SELLER_STATS_SELECT = """
WITH items AS (
    SELECT oi.order_id, oi.seller_id, p.product_category_name_english AS category, oi.price
    FROM main.order_items oi
    JOIN main.products p ON p.product_id = oi.product_id
    JOIN main.orders o   ON o.order_id = oi.order_id
    WHERE o.is_delivered = 1 AND o.in_trend_period = 1
),
sales AS (
    SELECT seller_id, category, COUNT(*) AS items_sold, ROUND(AVG(price), 2) AS avg_price
    FROM items GROUP BY seller_id, category
),
reviews AS (
    SELECT x.seller_id, x.category, COUNT(*) AS review_count, ROUND(AVG(x.review_score), 3) AS avg_review_score
    FROM (SELECT DISTINCT i.order_id, i.seller_id, i.category, r.review_score
          FROM items i JOIN main.order_reviews r ON r.order_id = i.order_id) x
    GROUP BY x.seller_id, x.category
)
SELECT s.seller_id, se.seller_city, se.seller_state, s.category, s.items_sold, s.avg_price,
       COALESCE(r.review_count, 0) AS review_count, r.avg_review_score
FROM sales s
JOIN main.sellers se ON se.seller_id = s.seller_id
LEFT JOIN reviews r ON r.seller_id = s.seller_id AND r.category = s.category
"""
SELLER_STATS_VIEW = "CREATE TEMP VIEW seller_stats AS" + SELLER_STATS_SELECT
SELLER_STATS_NOTE = (
    "VIEW seller_stats (platform-wide per seller and category, delivered orders in 2017-01..2018-08, one review per "
    "order): seller_id TEXT, seller_city TEXT, seller_state TEXT, category TEXT, items_sold INTEGER, avg_price REAL, "
    "review_count INTEGER, avg_review_score REAL"
)

PUBLIC_VIEWS = [CATEGORY_STATS_VIEW, PRODUCT_STATS_VIEW, STATE_DELIVERY_VIEW, SELLER_STATS_VIEW]
PUBLIC_VIEW_NOTES = [CATEGORY_STATS_NOTE, PRODUCT_STATS_NOTE, STATE_DELIVERY_NOTE, SELLER_STATS_NOTE]

# Versi view untuk customer (sebagian juga untuk seller) tanpa angka volume persis.
# Alasannya: items_sold x avg_price yang persis sama dengan omzet platform (selisih hanya ~R$60).
# Rentang kasar saja juga tidak cukup: dengan titik tengah rentang, jumlah dari ~70 kategori tetap memberi ~90%
# omzet asli (galatnya saling menutupi), dan review_count persis x harga memberi ~95%.
# Jadi penjualan hanya diberikan sebagai urutan (sales_rank), jumlah ulasan sebagai label yang dibatasi
# ('100 or more'), dan harga sebagai rentang (price_band) + urutan (price_level).
PRICE_LEVEL = """CASE WHEN avg_price < 25 THEN 1 WHEN avg_price < 50 THEN 2 WHEN avg_price < 100 THEN 3
    WHEN avg_price < 200 THEN 4 WHEN avg_price < 500 THEN 5 ELSE 6 END"""
PRICE_BAND = """CASE WHEN avg_price < 25 THEN 'under R$25' WHEN avg_price < 50 THEN 'R$25-49'
    WHEN avg_price < 100 THEN 'R$50-99' WHEN avg_price < 200 THEN 'R$100-199'
    WHEN avg_price < 500 THEN 'R$200-499' ELSE 'R$500 or more' END"""
REVIEW_VOLUME = """CASE WHEN review_count < 30 THEN 'under 30' WHEN review_count < 100 THEN '30-99'
    ELSE '100 or more' END"""
CUSTOMER_COLUMNS = (f"RANK() OVER (ORDER BY items_sold DESC) AS sales_rank, {PRICE_LEVEL} AS price_level, "
                    f"{PRICE_BAND} AS price_band, {REVIEW_VOLUME} AS review_volume, avg_review_score")
CUSTOMER_COLUMNS_NOTE = (
    "sales_rank INTEGER (1 = most items sold), price_level INTEGER (1 = cheapest band .. 6), price_band TEXT "
    "(average item price, e.g. 'R$50-99'), review_volume TEXT ('under 30', '30-99' or '100 or more'), "
    "avg_review_score REAL. Exact items sold, prices and review counts are not available to customers: rank with "
    "sales_rank / price_level, report price_band and review_volume, and for rating rankings use only "
    "review_volume <> 'under 30'."
)
CUSTOMER_CATEGORY_STATS_VIEW = (f"CREATE TEMP VIEW category_stats AS SELECT category, {CUSTOMER_COLUMNS} "
                                f"FROM ({CATEGORY_STATS_SELECT})")
CUSTOMER_CATEGORY_STATS_NOTE = ("VIEW category_stats (platform-wide, delivered orders in 2017-01..2018-08, one "
                                "review per order): category TEXT, " + CUSTOMER_COLUMNS_NOTE)
CUSTOMER_PRODUCT_STATS_VIEW = (f"CREATE TEMP VIEW product_stats AS SELECT product_id, category, {CUSTOMER_COLUMNS} "
                               f"FROM ({PRODUCT_STATS_SELECT})")
CUSTOMER_PRODUCT_STATS_NOTE = ("VIEW product_stats (platform-wide per product, delivered orders in 2017-01..2018-08, "
                               "one review per order): product_id TEXT, category TEXT, " + CUSTOMER_COLUMNS_NOTE)
# sales_rank seller dihitung per kategori ("seller nomor 1 di health_beauty"), bukan se-platform.
SELLER_COLUMNS = CUSTOMER_COLUMNS.replace("OVER (ORDER BY", "OVER (PARTITION BY category ORDER BY")
CUSTOMER_SELLER_STATS_VIEW = (f"CREATE TEMP VIEW seller_stats AS SELECT seller_id, seller_city, seller_state, category, "
                              f"{SELLER_COLUMNS} FROM ({SELLER_STATS_SELECT})")
CUSTOMER_SELLER_STATS_NOTE = ("VIEW seller_stats (platform-wide per seller and category, delivered orders in "
                              "2017-01..2018-08, one review per order): seller_id TEXT, seller_city TEXT, "
                              "seller_state TEXT, category TEXT, " + CUSTOMER_COLUMNS_NOTE.replace(
                                  "(1 = most items sold)", "(1 = most items sold within the category)"))
CUSTOMER_VIEWS = [CUSTOMER_CATEGORY_STATS_VIEW, CUSTOMER_PRODUCT_STATS_VIEW, STATE_DELIVERY_VIEW,
                  CUSTOMER_SELLER_STATS_VIEW]
CUSTOMER_VIEW_NOTES = [CUSTOMER_CATEGORY_STATS_NOTE, CUSTOMER_PRODUCT_STATS_NOTE, STATE_DELIVERY_NOTE,
                       CUSTOMER_SELLER_STATS_NOTE]
# Seller: product_stats dan seller_stats sama dengan versi customer (tanpa omzet seller lain), tetapi
# category_stats tetap memakai angka persis sebagai benchmark kategori.
SELLER_VIEWS = [CATEGORY_STATS_VIEW] + CUSTOMER_VIEWS[1:]
SELLER_VIEW_NOTES = [CATEGORY_STATS_NOTE] + CUSTOMER_VIEW_NOTES[1:]
# View dan deskripsinya (untuk prompt LLM) per role
VIEWS = {"company": PUBLIC_VIEWS, "seller": SELLER_VIEWS, "customer": CUSTOMER_VIEWS}
VIEW_NOTES = {"company": PUBLIC_VIEW_NOTES, "seller": SELLER_VIEW_NOTES, "customer": CUSTOMER_VIEW_NOTES}

# Kunci acak per proses untuk pseudonim pembeli (dipakai di view seller). Kuncinya sama selama aplikasi berjalan,
# jadi COUNT DISTINCT dan JOIN tetap benar; pseudonim tidak bisa dibalik ke id asli.
_PSEUDONYM_KEY = secrets.token_bytes(16)


def _pseudonym(value):
    """Ubah customer id menjadi pseudonim 'cust_' + 12 karakter hex (HMAC-SHA256). None tetap None."""
    if value is None:
        return None
    return "cust_" + hmac.new(_PSEUDONYM_KEY, str(value).encode(), hashlib.sha256).hexdigest()[:12]


@dataclass(frozen=True)
class Scope:
    """Role dan id pengguna yang sudah divalidasi (dibuat lewat make_scope)."""
    role: str = "company"
    user_id: str | None = None

    @property
    def is_restricted(self):
        """True untuk seller dan customer, yaitu role yang datanya dibatasi."""
        return self.role in ("seller", "customer")


# =============================================================================
# KONEKSI & SKEMA
# =============================================================================
def get_connection():
    """Buka koneksi read-only ke olist.db. Kalau file database belum ada, raise FileNotFoundError."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found. Run: python dataset/preprocessing/numeric_data/clean_olist.py")
    return sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)


@lru_cache(maxsize=3)
def get_schema(role="company"):
    """Buat teks skema (tabel, kolom, tipe, catatan kolom, dan view) untuk prompt LLM; hasilnya di-cache per role.
    Untuk role "customer" (anonim) hanya tabel umum + view agregat yang ditampilkan; tabel transaksi tidak disebut."""
    public_only = role == "customer"
    conn = get_connection()
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        if public_only:
            tables = [table for table in tables if table in PUBLIC_TABLES]
        lines = []
        for table in tables:
            cols = []
            for _, name, ctype, *_ in conn.execute(f'PRAGMA table_info("{table}")'):
                note = COLUMN_NOTES.get(f"{table}.{name}")
                cols.append(f"  {name} {ctype}" + (f"  -- {note}" if note else ""))
            lines.append(f"TABLE {table}:\n" + "\n".join(cols))
        lines += VIEW_NOTES[role]
        return "\n\n".join(lines)
    finally:
        conn.close()


# =============================================================================
# SCOPE PER ROLE
# =============================================================================
def make_scope(role="company", user_id=None):
    """Validasi role dan id, lalu kembalikan Scope.
    - company : tidak butuh id.
    - customer: pembeli anonim; id diabaikan kalau dikirim, jadi scope-nya sama untuk semua customer.
    - seller  : WAJIB punya id yang formatnya valid dan ada di tabel sellers.
    Raise ValueError kalau role atau id tidak valid."""
    role = (role or "company").strip().lower()
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}; use one of {ROLES}.")
    if role == "company":
        return Scope("company")
    if role == "customer":
        return Scope("customer")
    user_id = (user_id or "").strip().lower()
    if not user_id:
        raise ValueError(f"A {role} id is required for the {role} role.")
    if not ID_PATTERN.fullmatch(user_id):      # cek format dulu, karena id ini nanti ditulis langsung ke teks SQL view
        raise ValueError(f"Invalid {role} id: {user_id!r}")
    conn = get_connection()
    try:
        found = conn.execute("SELECT seller_id FROM sellers WHERE seller_id = ? LIMIT 1", (user_id,)).fetchone()
    finally:
        conn.close()
    if not found:
        raise ValueError(f"Unknown {role} id: {user_id}")
    return Scope(role, found[0])


def _scope_views(scope):
    """Buat daftar perintah CREATE TEMP VIEW untuk scope seller.
    Id sudah lolos ID_PATTERN (hanya karakter hex), jadi aman ditulis langsung ke teks SQL.
    Role lain mendapat list kosong: company membaca tabel asli, sedangkan customer tidak mendapat view transaksi
    (authorizer menolak tabel itu, jadi yang terbaca hanya tabel umum dan CUSTOMER_VIEWS)."""
    uid = scope.user_id
    if scope.role == "seller":
        own_orders = f"SELECT order_id FROM main.order_items WHERE seller_id = '{uid}'"
        # Nilai uang per order dihitung ulang dari barang seller ini saja (order bisa berisi barang seller lain)
        own_totals = ("SELECT order_id, ROUND(SUM(price), 2) AS items_value, "
                      "ROUND(SUM(freight_value), 2) AS freight_total, COUNT(*) AS n_items, "
                      "ROUND(SUM(price) + SUM(freight_value), 2) AS order_total "
                      f"FROM main.order_items WHERE seller_id = '{uid}' GROUP BY order_id")
        # kolom orders tanpa kolom uang (diganti versi seller di atas); customer_id diganti pseudonim
        order_cols = ", ".join("pseudonym(o.customer_id) AS customer_id" if c == "customer_id" else f"o.{c}"
                               for c in _order_columns() if c not in ORDER_MONEY_COLUMNS)
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
            # Pembeli tanpa id asli dan tanpa kode pos; kota & state tetap ada untuk agregat
            "CREATE TEMP VIEW customers AS SELECT pseudonym(customer_id) AS customer_id, "
            "pseudonym(customer_unique_id) AS customer_unique_id, customer_city, customer_state "
            "FROM main.customers WHERE customer_id IN "
            f"(SELECT customer_id FROM main.orders WHERE order_id IN ({own_orders}))",
        ]
    return []


# Kolom uang di tabel orders yang dihitung ulang khusus untuk seller
ORDER_MONEY_COLUMNS = ("items_value", "freight_total", "n_items", "order_total")


@lru_cache(maxsize=1)
def _order_columns():
    """Ambil nama kolom tabel orders (dipakai untuk membangun view orders seller); hasilnya di-cache."""
    conn = get_connection()
    try:
        return tuple(row[1] for row in conn.execute("PRAGMA table_info(orders)"))
    finally:
        conn.close()


def _authorizer(scope, temp_names=frozenset()):
    """Buat fungsi authorizer SQLite yang hanya mengizinkan aksi membaca.
    Untuk seller/customer, tabel asli (main) hanya boleh dibaca lewat view scope, kecuali PUBLIC_TABLES.
    temp_names = nama TEMP VIEW yang dibuat untuk koneksi ini (boleh dibaca)."""
    def check(action, arg1, arg2, db_name, source):
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            if arg1 and arg1.lower().startswith("sqlite_"):      # tabel internal SQLite (sqlite_master, dst.)
                return sqlite3.SQLITE_DENY
            if scope.is_restricted and source is None and arg1 not in PUBLIC_TABLES:
                # Membaca tabel asli secara langsung = mencoba melewati scope. COUNT(*) tanpa kolom dilaporkan
                # SQLite dengan db_name None (bukan "main"), jadi kasus itu juga dicek terhadap daftar view temp.
                if db_name == "main" or (db_name is None and arg1 not in temp_names):
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY               # aksi selain membaca: PRAGMA, ATTACH, menulis, membuat objek, dst.
    return check


# =============================================================================
# VALIDASI & EKSEKUSI
# =============================================================================
def _strip_literals(sql):
    """Kosongkan isi string '...' dan buang komentar SQL, supaya pengecekan kata kunci tidak salah tangkap
    (mis. kata 'DROP' di dalam teks)."""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def validate_query(sql):
    """Periksa SQL dari LLM. Kembalikan SQL yang sudah dirapikan (tanpa ';' di akhir),
    atau raise ValueError kalau SQL tidak boleh dijalankan."""
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
    """Jalankan SQL dari LLM dengan semua lapisan keamanan dan kembalikan hasilnya sebagai DataFrame.
    Urutan: validasi SQL -> koneksi read-only -> buat TEMP VIEW sesuai role -> pasang authorizer dan batas waktu
    -> jalankan query dan ambil maksimal max_rows baris.
    Raise error kalau SQL ditolak, gagal, atau melewati batas waktu."""
    sql = validate_query(sql)
    scope = scope or Scope("company")
    conn = get_connection()
    try:
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)   # cegah zeroblob/teks raksasa di memori
        conn.create_function("pseudonym", 1, _pseudonym, deterministic=True)
        view_sqls = _scope_views(scope) + VIEWS[scope.role]      # view scope seller + view agregat sesuai role
        for view_sql in view_sqls:
            conn.execute(view_sql)
        # nama semua view temp, supaya authorizer tahu view mana yang boleh dibaca
        temp_names = frozenset(re.search(r"CREATE TEMP VIEW (\w+)", v).group(1) for v in view_sqls)
        conn.set_authorizer(_authorizer(scope, temp_names))
        # progress handler dipanggil berkala; nilai 1 = hentikan query karena melewati deadline
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
    """Pendapatan bulanan (order delivered, 2017-01..2018-08), sudah dibatasi sesuai role.
    Kolom hasil: month, revenue, orders."""
    return run_query(MONTHLY_REVENUE_SQL, scope=make_scope(role, user_id), max_rows=1000)
