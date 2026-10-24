"""
test_sql_security.py - Tes OFFLINE (tanpa OpenAI) untuk sql_tools.py dan alur sql_agent.py.

Menguji: validasi SQL, scope per role, tool per sesi login, authorizer, batas baris/waktu,
angka referensi fitur Seller P0, dan alur ask_sql_agent dengan LLM palsu.

Jalankan dari folder utama proyek:
    python -m dataset.preprocessing.numeric_data.test_sql_security
"""
import inspect

from dataset.preprocessing.numeric_data.sql_tools import get_monthly_revenue, make_scope, run_query, validate_query

SELLER = "4869f7a5dfa277a7dca6462dcf3b52b2"        # seller dengan revenue tertinggi
OTHER_SELLER = "53243585a1d6dc2643021fd1853d8905"  # seller lain
REAL_CUSTOMER = "8d50f5eadf50201ccdcedfb9e2ac8455" # customer_unique_id asli (tidak boleh terlihat oleh seller)
PRIVATE_TABLES = ["orders", "order_items", "order_payments", "order_reviews", "customers", "sellers"]

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
check("customer anonim: tanpa id boleh", make_scope("customer").user_id is None)
check("customer anonim: id yang dikirim diabaikan", make_scope("customer", REAL_CUSTOMER).user_id is None)
check("id dengan SQL injection ditolak", raises(make_scope, "seller", "x' OR 1=1 --") is not None)
check("id yang tidak ada ditolak", raises(make_scope, "seller", "0" * 32) is not None)
check("role tidak dikenal ditolak", raises(make_scope, "admin") is not None)

seller = make_scope("seller", SELLER)
customer = make_scope("customer")

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

# customer anonim (aturan C1-C5 di agent/policy.py): tabel transaksi tertutup, hanya agregat umum
for table in PRIVATE_TABLES:
    check(f"customer: SELECT kolom {table} ditolak", raises(run_query, f"SELECT * FROM {table} LIMIT 1", customer))
    # COUNT(*) tidak membaca kolom; SQLite melaporkannya dengan db_name None (celah lama di authorizer)
    check(f"customer: COUNT(*) {table} ditolak", raises(run_query, f"SELECT COUNT(*) FROM {table}", customer))
check("customer: subquery ke tabel tertutup ditolak", raises(
    run_query, "SELECT category FROM category_stats WHERE category IN (SELECT seller_id FROM sellers)", customer))
check("customer: products & br_states tetap terbuka", value("SELECT COUNT(*) FROM products", customer) == 32951
      and value("SELECT COUNT(*) FROM br_states", customer) == 27)
check("customer: category_stats terbuka", value("SELECT COUNT(*) FROM category_stats", customer) > 50)
# C5: items_sold x avg_price yang persis = omzet platform; customer hanya dapat rentang kasar
check("customer: items_sold/avg_price/review_count persis tidak ada",
      all(raises(run_query, f"SELECT SUM({col}) FROM {view}", customer) is not None
          for col in ("items_sold", "avg_price", "review_count") for view in ("category_stats", "product_stats")))
check("customer: urutan penjualan tetap ada", run_query(
    "SELECT category FROM category_stats WHERE sales_rank = 1", customer).iloc[0, 0] == "bed_bath_table")
check("customer: product_stats terbuka", value("SELECT COUNT(*) FROM product_stats", customer) > 30_000)
check("customer: state_delivery_stats terbuka", value("SELECT COUNT(*) FROM state_delivery_stats", customer) == 27)

# seller: pembeli tanpa id asli & tanpa kode pos (aturan S3), agregat tetap benar
cols = list(run_query("SELECT * FROM customers LIMIT 1", seller).columns)
check("seller: customers tanpa kode pos", "customer_zip_code_prefix" not in cols, cols)
ids = run_query("SELECT customer_id, customer_unique_id FROM customers LIMIT 5", seller)
check("seller: id pembeli berupa pseudonim", all(v.startswith("cust_") for v in ids.values.ravel()), ids.values.tolist())
check("seller: id pembeli asli tidak bisa dicari",
      value(f"SELECT COUNT(*) FROM customers WHERE customer_unique_id = '{REAL_CUSTOMER}'", seller) == 0)
check("seller: pseudonim cocok di JOIN orders-customers", value(
    "SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id", seller)
      == value("SELECT COUNT(*) FROM orders", seller))
true_buyers = value(f"""SELECT COUNT(DISTINCT c.customer_unique_id) FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.order_id IN (SELECT order_id FROM order_items WHERE seller_id = '{SELLER}')""")   # company scope
check("seller: jumlah pembeli unik sama dengan data asli",
      value("SELECT COUNT(DISTINCT customer_unique_id) FROM customers", seller) == true_buyers, true_buyers)

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
    from agent.sql_agent import ask_sql_agent, make_data_tools, period_note
    from dataset.preprocessing.numeric_data.sql_tools import get_schema
except ImportError as error:
    print(f"[SKIP] tes agent: {error}")
else:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import Field

    class FakeLLM(BaseChatModel):
        """Chat model palsu untuk agent LangChain: mengembalikan jawaban berurutan dan mencatat prompt yang
        diterima. Jawaban berupa teks biasa, atau AIMessage dengan tool_calls (lihat sql_call)."""
        replies: list
        prompts: list = Field(default_factory=list)

        @property
        def _llm_type(self):
            return "fake"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.prompts.append(messages)
            reply = self.replies.pop(0)
            message = reply if isinstance(reply, AIMessage) else AIMessage(content=reply)
            return ChatResult(generations=[ChatGeneration(message=message)])

    call_ids = iter(range(1000))

    def sql_call(sql):
        """Balasan LLM yang memanggil tool run_sql dengan SQL ini (id tool call harus unik)."""
        return AIMessage(content="", tool_calls=[{"name": "run_sql", "args": {"query": sql}, "id": f"call_{next(call_ids)}"}])

    llm = FakeLLM(replies=[sql_call("SELECT ROUND(SUM(oi.price),2) AS revenue FROM order_items oi "
                                    "JOIN orders o ON o.order_id = oi.order_id WHERE o.is_delivered = 1"),
                           "Total pendapatan: R$ 226.987,93"])
    out = ask_sql_agent("Berapa total pendapatan semua seller?", role="seller", user_id=SELLER,
                        llm=llm, verbose=False, language="Indonesian")
    check("agent: SQL LLM dijalankan dengan scope seller", out["data"].iloc[0, 0] == 226987.93)
    check("agent: jawaban akhir dari LLM", out["answer"] == "Total pendapatan: R$ 226.987,93")

    llm = FakeLLM(replies=[sql_call('SELECT COUNT(*) FROM "main"."order_items"'),
                           sql_call("SELECT COUNT(*) FROM order_items"), "ok"])
    out = ask_sql_agent("count items", role="seller", user_id=SELLER, llm=llm, verbose=False, language="English")
    check("agent: SQL bypass ditolak lalu diperbaiki LLM", out["data"].iloc[0, 0] < 112650 and out["error"] is None)

    llm = FakeLLM(replies=[sql_call("SELECT nope FROM nowhere") for _ in range(3)] + ["I could not do it."])
    out = ask_sql_agent("broken", llm=llm, verbose=False, language="English")
    check("agent: berhenti setelah 3 SQL gagal", out["error"] and out["data"] is None and not llm.replies,
          out["error"])

    llm = FakeLLM(replies=["Sorry, the database has no profit data."])
    out = ask_sql_agent("What is my profit?", llm=llm, verbose=False, language="English")
    check("agent: pertanyaan di luar database -> jawab tanpa SQL", out["sql"] is None and out["error"] is None
          and "profit" in out["answer"])

    llm = FakeLLM(replies=["You need to log in as a seller first."])
    out = ask_sql_agent("my revenue?", role="seller", llm=llm, verbose=False, language="English")
    check("agent: seller tanpa id -> menolak tanpa menjalankan SQL", out["sql"] is None and out["error"])

    # catatan periode mengikuti SQL yang benar-benar dijalankan
    check("periode: SQL dengan in_trend_period -> 'January 2017 to August 2018'",
          "January 2017 to August 2018" in period_note("... WHERE o.in_trend_period = 1"))
    check("periode: SQL tanpa filter -> 'all available data'",
          "all available data" in period_note("SELECT seller_city, SUM(price) FROM order_items GROUP BY 1"))
    check("periode: SQL dengan periode dari pengguna -> catatan parsial",
          "partial" in period_note("... WHERE o.purchase_year = 2016"))

    llm = FakeLLM(replies=[sql_call("SELECT s.seller_city, ROUND(SUM(oi.price),2) AS revenue FROM order_items oi "
                                    "JOIN sellers s ON s.seller_id = oi.seller_id GROUP BY 1 ORDER BY 2 DESC LIMIT 3"),
                           "ok"])
    ask_sql_agent("Top 3 seller cities by revenue?", llm=llm, verbose=False, language="English")
    last_prompt = llm.prompts[-1]               # prompt saat LLM menulis jawaban akhir
    tool_result = next(m.content for m in last_prompt if isinstance(m, ToolMessage))
    check("agent: hasil tanpa filter periode tidak mengklaim 2017-2018",
          "all available data" in tool_result and "January 2017 to August 2018 (the default" not in tool_result)
    check("agent: instruksi meminta nama kota disalin persis", "EXACTLY" in last_prompt[0].content)

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

    customer_tools = make_data_tools("customer")
    check("tool customer anonim: hanya ask_data_agent", [tool_name(t) for t in customer_tools] == ["ask_data_agent"])
    public = get_schema(public_only=True)
    check("skema customer: tanpa tabel transaksi",
          all(f"TABLE {t}:" not in public for t in PRIVATE_TABLES) and "product_stats" in public, public[:80])

    llm = FakeLLM(replies=[sql_call("SELECT COUNT(*) FROM orders"),
                           sql_call("SELECT COUNT(*) FROM category_stats"), "ok"])
    out = ask_sql_agent("how many orders?", role="customer", llm=llm, verbose=False, language="English")
    check("agent customer: tabel orders ditolak, LLM pindah ke agregat umum",
          out["data"] is not None and out["sql"].endswith("FROM category_stats") and out["error"] is None)

print(f"\n{sum(results)}/{len(results)} PASS")
raise SystemExit(0 if all(results) else 1)
