"""
test_sql_security.py - Tes OFFLINE (tanpa OpenAI) untuk sql_tools.py dan alur sql_agent.py.

Menguji: validasi SQL, scope per role, tool per sesi login, authorizer, batas baris/waktu,
angka referensi fitur Seller P0, dan alur ask_sql_agent dengan LLM palsu.

Jalankan dari folder utama proyek:
    python -m data_preprocessing.numeric_data.test_sql_security
"""
import inspect
from types import SimpleNamespace

from data_preprocessing.numeric_data.sql_tools import get_monthly_revenue, make_scope, run_query, validate_query

SELLER = "4869f7a5dfa277a7dca6462dcf3b52b2"        # seller dengan revenue tertinggi
OTHER_SELLER = "53243585a1d6dc2643021fd1853d8905"  # seller lain
CUSTOMER = "8d50f5eadf50201ccdcedfb9e2ac8455"      # customer dengan order terbanyak

REVENUE = ("SELECT ROUND(SUM(oi.price), 2) AS revenue, COUNT(DISTINCT oi.seller_id) AS sellers "
           "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
           "WHERE o.is_delivered = 1 AND o.in_trend_period = 1")

results = []


def check(name, condition, detail=""):
    condition = bool(condition)
    results.append(condition)
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def raises(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def value(sql, scope=None):
    return run_query(sql, scope=scope).iloc[0, 0]


# ---------------------------------------------------------------- 1. validasi SQL
for bad in ["DELETE FROM orders", "SELECT 1; SELECT 2", "PRAGMA table_info(orders)",
            "SELECT * FROM main.order_items", "SELECT * FROM sqlite_master",
            "ATTACH DATABASE 'x.db' AS x", "WITH x AS (SELECT 1) INSERT INTO orders SELECT * FROM x"]:
    check(f"ditolak: {bad}", raises(validate_query, bad) is not None)
check("string berisi ';' dan 'DROP' tetap boleh", validate_query("SELECT 'a;b DROP' AS x;") == "SELECT 'a;b DROP' AS x")
check("fungsi replace() tetap boleh", value("SELECT replace('ab','a','c')") == "cb")

# ---------------------------------------------------------------- 2. scope / login
check("seller tanpa id ditolak", raises(make_scope, "seller") is not None)
check("customer tanpa id ditolak", raises(make_scope, "customer", "") is not None)
check("id dengan SQL injection ditolak", raises(make_scope, "seller", "x' OR 1=1 --") is not None)
check("id yang tidak ada ditolak", raises(make_scope, "seller", "0" * 32) is not None)
check("role tidak dikenal ditolak", raises(make_scope, "admin") is not None)

seller = make_scope("seller", SELLER)
customer = make_scope("customer", CUSTOMER)

company_rev = run_query(REVENUE).iloc[0]
check("company: revenue semua seller (2017-01..2018-08)", company_rev.revenue == 13179777.95, company_rev.revenue)

seller_rev = run_query(REVENUE, seller).iloc[0]
check("seller: 'revenue semua seller' hanya revenue sendiri",
      seller_rev.revenue == 226987.93 and seller_rev.sellers == 1, seller_rev.revenue)
check("seller: tidak bisa melihat seller lain",
      run_query(f"SELECT COUNT(*) FROM order_items WHERE seller_id = '{OTHER_SELLER}'", seller).iloc[0, 0] == 0)
check("seller: tabel sellers hanya berisi dirinya", value("SELECT COUNT(*) FROM sellers", seller) == 1)
check("seller: products tetap lengkap (data produk umum)", value("SELECT COUNT(*) FROM products", seller) == 32951)
check("seller: category_stats se-platform", value("SELECT SUM(items_sold) FROM category_stats", seller) > 100_000)

bypass = raises(run_query, 'SELECT COUNT(*) FROM "main"."order_items"', seller)
check("seller: bypass lewat \"main\".\"order_items\" ditolak authorizer", bypass is not None, bypass)
check("company: \"main\".\"order_items\" boleh", value('SELECT COUNT(*) FROM "main"."order_items"') == 112650)

check("customer: hanya order miliknya",
      value("SELECT COUNT(DISTINCT customer_unique_id) FROM customers", customer) == 1)
check("customer: order_items hanya dari ordernya",
      value("SELECT COUNT(*) FROM order_items oi LEFT JOIN orders o ON o.order_id = oi.order_id "
            "WHERE o.order_id IS NULL", customer) == 0)

# ---------------------------------------------------------------- 3. batas baris & waktu
cats = run_query("SELECT DISTINCT product_category_name_english FROM products")
check("hasil >50 baris dipotong dan ditandai", len(cats) == 50 and cats.attrs["truncated"])
check("query terlalu lama dihentikan",
      "TimeoutError" in (raises(run_query, "SELECT COUNT(*) FROM order_items a, order_items b", timeout=1) or ""))

big = raises(run_query, "SELECT length(zeroblob(500000000))")
check("nilai raksasa (zeroblob 500 MB) ditolak", big is not None, big)

# ---------------------------------------------------------------- 3b. uang seller di order bersama
own_items = value("SELECT ROUND(SUM(price), 2) FROM order_items", seller)
check("seller: SUM(orders.items_value) = barang sendiri saja",
      value("SELECT ROUND(SUM(items_value), 2) FROM orders", seller) == own_items == 229472.63, own_items)
own_total = value("SELECT SUM(order_total) FROM orders", seller)
paid = value("SELECT SUM(payment_value) FROM order_payments", seller)
check("seller: payment_value = porsi seller (bukan seluruh order)", abs(paid - own_total) / own_total < 0.02,
      f"bayar {paid:,.2f} vs order_total {own_total:,.2f}")

# ---------------------------------------------------------------- 3c. kota seller bersih
check("kota seller: São Paulo hanya satu ejaan",
      value("SELECT COUNT(DISTINCT seller_city) FROM sellers WHERE seller_city LIKE '%s_o paul%'") == 1)
check("kota seller: tidak ada nama aneh (angka, email, '/ sp')",
      value("SELECT COUNT(*) FROM sellers WHERE seller_city GLOB '*[^a-z'' ]*'") == 0)

# ---------------------------------------------------------------- 4. angka referensi Seller P0 (seller SELLER)
monthly = get_monthly_revenue("seller", SELLER)
check("income trend: 20 bulan max, total = revenue seller",
      len(monthly) <= 20 and round(monthly.revenue.sum(), 2) == 226987.93, f"{len(monthly)} bulan")

pay = run_query("""
    SELECT p.payment_type, COUNT(DISTINCT p.order_id) AS orders, ROUND(AVG(p.payment_installments), 2) AS avg_inst
    FROM order_payments p JOIN orders o ON o.order_id = p.order_id
    WHERE o.is_delivered = 1 AND o.in_trend_period = 1
    GROUP BY p.payment_type ORDER BY orders DESC""", seller)
check("customer behavior: jenis pembayaran", pay.payment_type.iloc[0] == "credit_card", pay.to_dict("records"))

repeat = run_query("""
    WITH c AS (SELECT cu.customer_unique_id, COUNT(DISTINCT o.order_id) AS n
               FROM orders o JOIN customers cu ON cu.customer_id = o.customer_id
               WHERE o.is_delivered = 1 AND o.in_trend_period = 1 GROUP BY cu.customer_unique_id)
    SELECT COUNT(*) AS buyers, SUM(n > 1) AS repeaters, ROUND(100.0 * SUM(n > 1) / COUNT(*), 2) AS pct FROM c""",
                   seller).iloc[0]
check("customer behavior: repeat purchase rate", repeat.buyers > 0, repeat.to_dict())

states = run_query("""
    SELECT s.state_name, COUNT(DISTINCT cu.customer_unique_id) AS buyers
    FROM orders o JOIN customers cu ON cu.customer_id = o.customer_id
    JOIN br_states s ON s.state_code = cu.customer_state
    WHERE o.is_delivered = 1 AND o.in_trend_period = 1
    GROUP BY s.state_name ORDER BY buyers DESC LIMIT 3""", seller)
check("customer behavior: sebaran state (nama lengkap)", states.state_name.iloc[0] == "São Paulo",
      states.to_dict("records"))

loved = run_query("""
    SELECT category, review_count, avg_review_score FROM category_stats
    WHERE review_count >= 30 ORDER BY avg_review_score DESC LIMIT 3""", seller)
check("most-loved categories (se-platform)", len(loved) == 3, loved.to_dict("records"))

# ---------------------------------------------------------------- 5. alur ask_sql_agent dengan LLM palsu
try:
    from agent.sql_agent import ask_sql_agent, get_customer_categories, make_data_tools, period_note
except ImportError as error:
    print(f"[SKIP] tes agent: {error}")
else:
    class FakeLLM:
        """Mengembalikan jawaban berurutan; mencatat prompt yang diterima."""
        def __init__(self, replies):
            self.replies, self.prompts = list(replies), []

        def invoke(self, messages):
            self.prompts.append(messages)
            return SimpleNamespace(content=self.replies.pop(0))

    llm = FakeLLM(["SELECT ROUND(SUM(oi.price),2) AS revenue FROM order_items oi "
                   "JOIN orders o ON o.order_id = oi.order_id WHERE o.is_delivered = 1",
                   "Total pendapatan: R$ 226.987,93"])
    out = ask_sql_agent("Berapa total pendapatan semua seller?", role="seller", user_id=SELLER,
                        llm=llm, verbose=False, language="Indonesian")
    check("agent: SQL LLM dijalankan dengan scope seller", out["data"].iloc[0, 0] == 226987.93)

    llm = FakeLLM(['SELECT COUNT(*) FROM "main"."order_items"', "SELECT COUNT(*) FROM order_items", "ok"])
    out = ask_sql_agent("count items", role="seller", user_id=SELLER, llm=llm, verbose=False, language="English")
    check("agent: SQL bypass ditolak lalu diperbaiki LLM", out["data"].iloc[0, 0] < 112650 and out["error"] is None)

    llm = FakeLLM(["You need to log in as a seller first."])
    out = ask_sql_agent("my revenue?", role="seller", llm=llm, verbose=False, language="English")
    check("agent: seller tanpa id -> menolak tanpa menjalankan SQL", out["sql"] is None and out["error"])

    # catatan periode mengikuti SQL yang benar-benar dijalankan
    check("periode: SQL dengan in_trend_period -> 'January 2017 to August 2018'",
          "January 2017 to August 2018" in period_note("... WHERE o.in_trend_period = 1"))
    check("periode: SQL tanpa filter -> 'all available data'",
          "all available data" in period_note("SELECT seller_city, SUM(price) FROM order_items GROUP BY 1"))
    check("periode: SQL dengan periode dari pengguna -> catatan parsial",
          "partial" in period_note("... WHERE o.purchase_year = 2016"))

    llm = FakeLLM(["SELECT s.seller_city, ROUND(SUM(oi.price),2) AS revenue FROM order_items oi "
                   "JOIN sellers s ON s.seller_id = oi.seller_id GROUP BY 1 ORDER BY 2 DESC LIMIT 3", "ok"])
    ask_sql_agent("Top 3 seller cities by revenue?", llm=llm, verbose=False, language="English")
    summary = llm.prompts[-1][0][1]
    check("agent: ringkasan tanpa filter periode tidak mengklaim 2017-2018",
          "all available data" in summary and "January 2017 to August 2018 (the default" not in summary)
    check("agent: ringkasan meminta nama kota disalin persis", "EXACTLY" in summary)

    # tool per sesi login: role & id dikunci, LLM hanya mengisi pertanyaan
    def tool_name(t):
        return getattr(t, "name", None) or t.__name__

    def tool_args(t):
        return set(t.args) if hasattr(t, "args") else set(inspect.signature(t).parameters)

    def call(t, **kwargs):
        return t.invoke(kwargs) if hasattr(t, "invoke") else t(**kwargs)

    seller_tools = make_data_tools("seller", SELLER)
    check("tool seller: hanya ask_data_agent", [tool_name(t) for t in seller_tools] == ["ask_data_agent"])
    check("tool: argumen hanya 'question' (tidak ada role/user_id)", tool_args(seller_tools[0]) == {"question"})
    check("tool: login seller tanpa id ditolak saat tool dibuat", raises(make_data_tools, "seller") is not None)

    customer_tools = {tool_name(t): t for t in make_data_tools("customer", CUSTOMER)}
    history = customer_tools.get("customer_past_categories")
    check("tool customer: customer_past_categories tanpa argumen", history is not None and tool_args(history) == set())
    text = call(history)
    check("tool customer: riwayat kategori miliknya sendiri", text.startswith("This customer bought before"), text)
    check("riwayat kategori: role selain customer ditolak",
          raises(get_customer_categories, make_scope("seller", SELLER)) is not None)
    check("riwayat kategori: hasil sama dengan query manual customer itu",
          get_customer_categories(customer).category.tolist()
          == run_query("SELECT p.product_category_name_english AS c, COUNT(*) n, MAX(o.order_purchase_timestamp) t "
                       "FROM orders o JOIN order_items oi ON oi.order_id = o.order_id "
                       "JOIN products p ON p.product_id = oi.product_id WHERE o.is_delivered = 1 "
                       "AND p.product_category_name_english != 'uncategorized' GROUP BY c "
                       "ORDER BY n DESC, t DESC LIMIT 3", customer).c.tolist())

print(f"\n{sum(results)}/{len(results)} PASS")
raise SystemExit(0 if all(results) else 1)
