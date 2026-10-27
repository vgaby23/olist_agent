"""
sql_agent.py - SQL Agent untuk Olist E-Commerce AI Assistant.

Agent ini menjawab pertanyaan yang butuh angka (pendapatan, jumlah, rata-rata, ranking, tren)
dengan menulis query SQL, menjalankannya ke database SQLite, lalu merangkum hasilnya.

Alur kerja untuk satu pertanyaan:
  1. make_scope memeriksa role (company / seller / customer) dan id pengguna.
  2. Dibuat agent LangChain dengan satu tool: run_sql(query).
  3. LLM menulis SQL lalu memanggil run_sql.
  4. run_sql menjalankan query lewat run_query: read-only dan dibatasi sesuai role
     (TEMP VIEW + authorizer di sql_tools.py).
  5. Kalau query error, pesan error dikirim kembali ke LLM untuk diperbaiki
     (maksimal MAX_RETRIES kali perbaikan).
  6. LLM menulis jawaban singkat berdasarkan hasil query.

Hal penting:
  - Pembatasan role ditegakkan di kode (make_scope dan run_query), tidak hanya lewat prompt.
  - Tool run_sql dibuat ulang untuk setiap pertanyaan dengan scope yang sudah dikunci (closure),
    jadi LLM hanya bisa mengisi teks SQL-nya.
  - Tool untuk supervisor dibuat sekali per sesi login lewat make_data_tools(role, user_id),
    jadi LLM tidak bisa memilih atau mengganti role.
  - Periode analisis default adalah 2017-01 s.d. 2018-08. Data di luar periode itu tidak lengkap,
    dan tahun 2018 hanya berisi Januari-Agustus.
  - Produk tanpa kategori diberi label 'uncategorized'.
  - View category_stats berisi ringkasan per kategori se-platform, untuk pembanding pasar.
"""
import contextvars
import os
import re
import threading
from typing import Annotated

from langchain.agents import create_agent
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.errors import GraphRecursionError

from agent.language import guess_language
from dataset.preprocessing.numeric_data.sql_tools import get_schema, make_scope, run_query

# Aksara non-Latin yang tidak boleh muncul di jawaban:
# Cyrillic, Ibrani, Arab, Devanagari, Thai, Hiragana/Katakana, Han (CJK), dan Hangul.
FOREIGN_SCRIPT = re.compile(r"[Ѐ-ӿ֐-ۿऀ-ॿ฀-๿぀-ヿ一-鿿가-힯]")

MAX_RETRIES = 2          # berapa kali LLM boleh memperbaiki SQL yang gagal
MAX_QUERIES = 4          # batas total query per pertanyaan (satu pertanyaan boleh dipecah jadi beberapa query)
RECURSION_LIMIT = 20     # batas langkah agent agar tidak berputar terus (1 panggilan LLM atau tool = 1 langkah)

# Terjemahan jenis pembayaran; hanya ditambahkan ke prompt kalau bahasa jawabannya Bahasa Indonesia
INDONESIAN_TERMS = ("- Payment types: write credit_card as 'kartu kredit', debit_card as 'kartu debit', "
                    "boleto as 'boleto (transfer bank Brasil)', voucher as 'voucher'.\n")

# System prompt utama: tugas agent, skema database ({schema} diisi get_schema), dan aturan bisnis untuk SQL
RULES = """
You are an expert SQLite analyst for the Olist Brazilian e-commerce database.
Answer the user's question with data: write a read-only SQLite query (SELECT or WITH ... SELECT), run it with
the run_sql tool, then answer from the result. Prefer ONE query; use a second one only if the question really
needs it. If run_sql returns an ERROR, fix the query and call run_sql again.

DATABASE SCHEMA:
{schema}

BUSINESS RULES (follow them strictly):
1. Revenue / sales / ranking by sales: only count delivered orders -> JOIN orders and filter
   orders.is_delivered = 1. Revenue = SUM(order_items.price) (no freight, no canceled or in-progress orders).
   Include freight_value only if the user asks for it. Never use order_payments.payment_value for revenue.
   The same delivered-only rule applies to counting buyers, repeat customers, payment-method usage and any other
   customer-behaviour statistic, unless the question is explicitly about order status (for example canceled orders).
2. Period: if the user does NOT name a period, ALWAYS JOIN orders and filter orders.in_trend_period = 1 (the
   default analysis window 2017-01 to 2018-08; do not write that range by hand). This also applies to rankings and
   totals by city, state, seller, category or payment type, not only to trends. If the user names a period, filter on
   orders.purchase_month ('YYYY-MM') or orders.purchase_year instead. Monthly trends: GROUP BY orders.purchase_month.
   Yearly: GROUP BY orders.purchase_year. Exception: when the user asks about the WHOLE dataset ("in the dataset",
   "in total", "all time", "di seluruh data"), use no period filter.
3. A customer is identified by customers.customer_unique_id (customer_id is unique per order, NOT per person).
   Join: orders.customer_id = customers.customer_id. When counting customers ALWAYS use
   COUNT(DISTINCT customers.customer_unique_id); never COUNT(*) or COUNT(customer_id) on the customers table.
   Repeat customers = customer_unique_id with COUNT(DISTINCT orders.order_id) > 1 among delivered orders.
   Customer state names: JOIN br_states ON br_states.state_code = customers.customer_state.
4. Product category: use products.product_category_name_english (the value 'uncategorized' means missing
   category). Products have NO name, only product_id and category.
5. Reviews: order_reviews has exactly one row per ORDER (scores only, no review text). Average score =
   AVG(order_reviews.review_score), joined through order_id. One order can have SEVERAL order_items rows, so for
   product, category or seller level scores first take ONE row per order in a CTE:
   SELECT DISTINCT oi.order_id, <product_id / category / seller_id>, r.review_score, then COUNT(*) / AVG over it.
   Never AVG or COUNT review_score directly over an order_items join.
   Review scores and review counts ALWAYS use delivered orders only: JOIN orders and filter orders.is_delivered = 1
   (canceled or undelivered orders are not part of the rating), together with the period filter from rule 2.
6. On-time delivery: orders.is_late is 1 (late) or 0 (on time) for delivered orders, NULL otherwise.
   On-time rate = 1 - AVG(is_late) over orders with is_delivered = 1. Delivery time = orders.delivery_days.
7. Payments: order_payments can have several rows per order. NEVER SUM values after joining order_items
   and order_payments together (rows multiply). Aggregate each table separately, or use orders.order_total.
   Installments: AVG(payment_installments), usually grouped by payment_type.
8. Sellers have NO name: identify them by seller_id plus seller_city and seller_state.
9. The database has NO cost, profit, or commission data. Product dimensions/weight are not available either.
10. Use LIMIT 10 for rankings and lists unless the user asks for a different number. Round money to 2 decimals.
    Only when the user asks for ONE item ("the top category", "my best-selling product"), still use LIMIT 5 so a
    tie is visible. Plural questions ("which categories...", "top products") get the normal LIMIT 10 list; never
    filter them down to rank 1.
11. Only use tables, views and columns that exist in the schema. Never prefix them with main. or temp.
12. When the answer is about specific sellers, orders, products or customers, ALWAYS also SELECT their id column
    (seller_id, order_id, product_id, customer_unique_id), so other agents can use it.
13. Ranking by AVERAGE score (most-loved, best or worst rated products, sellers or categories): only include items
    with at least 30 reviewed orders (HAVING COUNT(*) >= 30 on the one-row-per-order CTE) unless the user gives
    another minimum, and also show the number of reviews.
14. Customer spending (what a customer paid) = SUM(orders.order_total) over delivered orders (price + freight).
15. Column values: orders.order_status is one of delivered, shipped, canceled, unavailable, invoiced, processing,
    created, approved. order_payments.payment_type is one of credit_card, boleto, voucher, debit_card.
    Dates are TEXT 'YYYY-MM-DD HH:MM:SS'. Money is in BRL.
16. category_stats is a ready-made, platform-wide summary per product category (all sellers). Use it for
    market or competitor comparisons by category (most-loved categories on the platform, average price, items sold).
    product_stats is the same per product_id, seller_stats per seller_id and category (use it to recommend or
    rank sellers in a category, e.g. the best-rated sellers for make up), state_delivery_stats gives delivery
    times per buyer state.
17. Category names: always filter on the exact English category name (e.g. 'health_beauty'). The supervisor maps
    product types ("make up", "sepatu") to that name before asking; never invent a category name.

WHEN THE DATABASE CANNOT ANSWER:
- If the question cannot be answered with this database (for example profit, product name, seller name),
  do NOT call run_sql: politely say so in one or two short sentences and do not invent any data.
""".strip()

# Aturan menulis jawaban akhir (digabung ke system prompt agent); {language} diisi bahasa jawaban
ANSWER_RULES = """
WRITING THE ANSWER (after run_sql returned a result):
- Answer the user's question using ONLY the query result. Do not invent numbers. If the result is empty, say
  that no data was found.
- Write the whole answer in {language} only. Do not mix in any other language and do not use any non-Latin
  writing system.
- Be concise. Do not mention column names, table names or SQL.
- All money values are in Brazilian Real; write them as R$ with the usual number format of the answer
  language (Indonesian/Portuguese: 1.234,56; English: 1,234.56). Do not convert currencies.
- Sellers are identified by seller_id only (no names).
- Copy every id (seller_id, order_id, product_id, customer_unique_id) in FULL, exactly as in the result.
  Never shorten an id.
- Copy city names and state names EXACTLY as they appear in the result (same spelling; you may only capitalise
  them). Never replace them with a similar-looking name.
- Copy category names EXACTLY as in the result, underscores included and NOT capitalised: write health_beauty,
  never "Health & Beauty" or "Health Beauty" (other agents search with these exact names).
- Follow the NOTES FOR THE ANSWER that run_sql returns with each result (period covered, truncated rows).
  ALWAYS state the period the numbers cover in one short phrase, even for a single number.
- Ties: when the user asked for ONE item and several items share the deciding value, say there is a tie and
  name all of them; never present just one as the single winner. In a ranked list, just show tied ranks.
- When the answer reports revenue or sales value, say in a few words what it counts: item prices of delivered
  orders, excluding freight (or including freight, if the query added it).
""".strip()


def get_llm():
    """Buat model chat OpenAI lewat agent/llm.py (API key dan nama model dibaca dari file .env).
    temperature=0 supaya SQL yang dihasilkan konsisten. Model reasoning yang tidak mendukung
    temperature ditangani di dalam chat_model."""
    from agent.llm import chat_model

    return chat_model(temperature=0)


def _text(response):
    """Ambil teks dari respons LLM. content bisa berupa string atau list blok; hasilnya selalu satu string."""
    content = response.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content.strip()


def _clean_sql(text):
    """Hapus pembungkus markdown ```sql ... ``` kalau LLM menambahkannya, supaya SQL bisa dijalankan."""
    text = re.sub(r"^```(?:sql)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def pick_language(question, language=None):
    """Tentukan bahasa jawaban, dengan urutan prioritas:
    1. argumen language (biasanya dikirim supervisor),
    2. petunjuk di teks pertanyaan, mis. "Answer in Bahasa Indonesia." atau "ANSWER LANGUAGE: English.",
    3. tebakan dari kata-kata di pertanyaan (guess_language),
    4. kalau semua gagal: "the same language as the question"."""
    if language:
        return language
    # cocok dengan "Answer in Bahasa Indonesia." maupun "ANSWER LANGUAGE: Bahasa Indonesia."
    hint = re.search(r"answer (?:in|language\s*:)\s*([A-Za-z][A-Za-z ]{1,30}?)\s*[.!,\n]",
                     question + "\n", re.IGNORECASE)
    if hint:
        return hint.group(1).strip()
    return guess_language(question) or "the same language as the question"


def make_role_note(scope):
    """Buat catatan untuk LLM: siapa penggunanya dan data apa yang bisa ia lihat.
    Catatan ini hanya penjelasan untuk LLM; pembatasan datanya sendiri sudah ditegakkan di run_query."""
    if scope.role == "seller":
        return (f"The user is the seller {scope.user_id}. The tables sellers, order_items, orders, order_payments, "
                "order_reviews and customers ALREADY contain only this seller's own data, so 'my', 'me', 'I', "
                "'saya', 'toko saya' simply mean all rows; no seller filter is needed. Other sellers' data is not "
                "available. In orders, items_value/freight_total/n_items/order_total count only this seller's items, "
                "and order_payments.payment_value is this seller's share of each payment (orders can also contain "
                "other sellers' items). For revenue still use SUM(order_items.price). For comparisons with the "
                "market use category_stats (platform-wide, per category, exact). product_stats and seller_stats "
                "are platform-wide too but, like for shoppers, show sales_rank, price_band and review_volume "
                "instead of exact items sold, prices and review counts; never estimate other sellers' revenue.")
    if scope.role == "customer":
        return ("The user is an anonymous shopper (no account). Only the tables products and br_states and the "
                "platform-wide views category_stats, product_stats, seller_stats and state_delivery_stats can be "
                "read; every other table is blocked. Answer with these aggregates only. For shoppers, items sold "
                "and average price are not available: use sales_rank and price_level for ordering, report "
                "price_band and review_volume, and never estimate sales volume or revenue. Recommending or ranking "
                "sellers by rating or sales_rank with seller_stats is allowed: always select and show the full "
                "seller_id with seller_city and seller_state for each seller. Questions about one specific order "
                "or customer, a seller's revenue or company revenue cannot be answered: say so politely without "
                "calling run_sql.")
    return "The user is a company (platform) employee and can see all data."


# Pola untuk mendeteksi filter periode, tabel transaksi, dan view agregat di dalam teks SQL
PERIOD_FILTER = re.compile(r"in_trend_period|purchase_month|purchase_year|order_purchase_timestamp")
ORDER_TABLES = re.compile(r"\b(orders|order_items|order_payments|order_reviews)\b")
PUBLIC_VIEWS = re.compile(r"\b(category_stats|product_stats|seller_stats|state_delivery_stats)\b")


def missing_period(sql):
    """True kalau SQL membaca tabel order tetapi tidak memakai filter periode apa pun (aturan 2 di RULES terlewat)."""
    lowered = sql.lower()
    return bool(ORDER_TABLES.search(lowered)) and not PERIOD_FILTER.search(lowered)


def period_note(sql):
    """Buat catatan periode untuk jawaban, berdasarkan filter yang BENAR-BENAR ada di SQL (bukan asumsi):
    - in_trend_period atau view agregat -> Januari 2017 s.d. Agustus 2018
    - filter periode dari pengguna      -> ingatkan bahwa bulan di luar 2017-01..2018-08 datanya tidak lengkap
    - tanpa filter periode              -> seluruh data (September 2016 s.d. Oktober 2018)"""
    lowered = sql.lower()
    if "in_trend_period" in lowered or PUBLIC_VIEWS.search(lowered):    # view agregat juga memakai periode default
        note =("- The numbers cover January 2017 to August 2018 (the default analysis period); say so briefly. "
                "If years are compared or a 2018 total is given, mention that 2018 only contains January to August.")
    elif PERIOD_FILTER.search(lowered):
        note = ("- The query filters on the period the user asked for. If that period includes months before "
                "2017-01 or after 2018-08, say the data for those months is partial. If years are compared or a "
                "2018 total is given, mention that 2018 only contains January to August.")
    else:
        note = ("- The query has no period filter, so the numbers cover all available data (September 2016 to "
                "October 2018). Do not claim any narrower period.")
    return note + "\n"


def _say(llm, language, message, question):
    """Minta LLM menulis 1-2 kalimat sopan ke pengguna (mis. penolakan karena login tidak lengkap)
    dalam bahasa jawaban."""
    prompt = (f"Politely tell the user, in one or two short sentences written in {language}, "
              f"that you cannot answer it because: {message}. Do not invent any data.\n\nQuestion: {question}")
    return _text(llm.invoke([("human", prompt)]))


def format_result(data, sql):
    """Ubah hasil query menjadi teks untuk LLM: tabel hasil + bagian NOTES FOR THE ANSWER
    (periode data, baris yang terpotong, dan pengingat aturan 2 kalau filter periode terlupa)."""
    table = data.to_string(index=False) if not data.empty else "(no rows returned)"
    notes = period_note(sql)
    if data.attrs.get("truncated"):
        notes += f"- Only the first {len(data)} rows are shown; the full result has more. Tell the user.\n"
    if missing_period(sql):     # LLM sering lupa aturan 2, jadi pengingatnya juga ditaruh di hasil query
        notes += ("- CHECK RULE 2: this query has no period filter. If the user did NOT name a period and did not "
                  "ask about the whole dataset, run the query again with JOIN orders ... AND "
                  "orders.in_trend_period = 1, and answer from that result.\n")
    return f"{table}\n\nNOTES FOR THE ANSWER:\n{notes}"


def make_sql_tool(scope, state, verbose=False):
    """Buat tool run_sql untuk SATU pertanyaan.

    scope dikunci di dalam fungsi ini (closure), jadi LLM hanya bisa mengisi teks SQL-nya.

    state adalah dict yang dipakai bersama dengan ask_sql_agent:
      - state["runs"]    : semua query yang dijalankan, tiap item {"call_id", "sql", "data", "error"}
      - state["error"]   : pesan error terakhir (None kalau query terakhir berhasil)
      - state["queries"] : jumlah query yang sudah dijalankan (maks. MAX_QUERIES)
      - state["failed"]  : jumlah query yang gagal

    LLM bisa memanggil run_sql beberapa kali sekaligus, dan LangChain menjalankannya di thread terpisah
    dengan urutan yang tidak pasti. Karena itu penghitung dijaga dengan lock, dan tiap run menyimpan
    tool_call_id-nya supaya ask_sql_agent bisa mengurutkannya sesuai urutan panggilan LLM."""
    lock = threading.Lock()

    @tool
    def run_sql(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
        """Run ONE read-only SQLite query (SELECT or WITH ... SELECT) on the Olist database and return the result
        table, or an ERROR message to fix. The data is already limited to what the logged-in user may see.

        Arguments:
            query: the SQL query only, no explanation and no markdown fences.
        """
        sql = _clean_sql(query)
        with lock:
            # berhenti kalau sudah terlalu sering gagal atau batas jumlah query tercapai
            if state["failed"] > MAX_RETRIES or state["queries"] >= MAX_QUERIES:
                return "STOP: no more queries allowed for this question. Do not call run_sql again; answer now."
            state["queries"] += 1
            number = state["queries"]
            run = {"call_id": tool_call_id, "sql": sql, "data": None, "error": None}
            state["runs"].append(run)
        if verbose:
            print(f"[Percobaan {number}] SQL:\n{sql}\n")
        try:
            data = run_query(sql, scope=scope)
        except Exception as e:  # SQL salah / ditolak -> pesan error dikirim ke LLM supaya diperbaiki
            run["error"] = f"{type(e).__name__}: {e}"
            with lock:
                state["failed"] += 1
                state["error"] = run["error"]
            if verbose:
                print(f"   -> Error: {run['error']}\n")
            return f"ERROR: {run['error']}\nFix the query and call run_sql again."

        run["data"] = data
        with lock:
            state["error"] = None
        return format_result(data, sql)

    return run_sql


def build_sql_agent(llm, scope, language, state, verbose=False):
    """Gabungkan LLM, tool run_sql, dan system prompt (RULES + ANSWER_RULES) menjadi agent LangChain.
    Agent dibuat ulang untuk setiap pertanyaan karena scope dan bahasa jawaban bisa berbeda."""
    system_prompt = "\n\n".join([
        RULES.format(schema=get_schema(scope.role)),
        ANSWER_RULES.format(language=language) + ("\n" + INDONESIAN_TERMS if "indones" in language.lower() else ""),
    ])
    return create_agent(llm, [make_sql_tool(scope, state, verbose)], system_prompt=system_prompt, name="sql_agent")


def ask_sql_agent(question, role="company", llm=None, verbose=True, language=None, user_id=None):
    """Jawab satu pertanyaan berbasis data dan kembalikan hasilnya sebagai dict.

    Argumen:
        question : pertanyaan pengguna.
        role     : "company", "seller", atau "customer".
        llm      : model chat; kalau None, dibuat lewat get_llm().
        verbose  : True = cetak setiap SQL dan error ke terminal.
        language : bahasa jawaban; kalau None, ditentukan oleh pick_language.
        user_id  : seller_id, wajib untuk role "seller". Role "customer" adalah pembeli anonim,
                   jadi id tidak dipakai.

    Isi dict yang dikembalikan:
        question, role : salinan input.
        sql, data      : SQL dan DataFrame dari query TERAKHIR yang berhasil (urut sesuai panggilan LLM).
                         Kalau tidak ada yang berhasil: sql = query terakhir yang dicoba, data = None.
        results        : SEMUA query yang berhasil, [{"sql", "data"}, ...]
                         (untuk pertanyaan yang dipecah menjadi beberapa query).
        answer         : jawaban akhir untuk pengguna.
        error          : pesan error, atau None kalau berhasil.
    """
    llm = llm or get_llm()
    language = pick_language(question, language)
    result = {"question": question, "role": role, "sql": None, "data": None, "results": [], "error": None}

    try:
        scope = make_scope(role, user_id)      # validasi role + id; kalau gagal, tolak tanpa menjalankan SQL
    except ValueError as error:
        result["error"] = str(error)
        result["answer"] = _say(llm, language, f"the login is incomplete ({error})", question)
        return result

    state = {"runs": [], "error": None, "queries": 0, "failed": 0}
    agent = build_sql_agent(llm, scope, language, state, verbose)
    # Catatan role ditaruh di pesan pengguna, tepat sebelum pertanyaan. Kalau ditaruh di system prompt, LLM
    # cenderung menambahkan filter id sendiri (mis. customer_id = <id>) sehingga hasilnya kosong.
    message = f"{make_role_note(scope)}\nQuestion: {question}"
    try:
        output = agent.invoke({"messages": [{"role": "user", "content": message}]},
                              config={"recursion_limit": RECURSION_LIMIT})
        answer = _text(output["messages"][-1])
        # urutkan run sesuai urutan tool call di pesan LLM (bukan urutan selesainya thread)
        call_order = [call["id"] for msg in output["messages"] for call in getattr(msg, "tool_calls", None) or []]
        state["runs"].sort(key=lambda run: call_order.index(run["call_id"]) if run["call_id"] in call_order
                           else len(call_order))
    except GraphRecursionError:            # agent melewati RECURSION_LIMIT langkah
        state["error"] = state["error"] or "the agent needed too many steps"
        answer = None

    succeeded = [{"sql": run["sql"], "data": run["data"]} for run in state["runs"] if run["data"] is not None]
    result["results"] = succeeded
    if succeeded:
        result.update(sql=succeeded[-1]["sql"], data=succeeded[-1]["data"])
    elif state["runs"]:
        result["sql"] = state["runs"][-1]["sql"]
    if not succeeded and state["error"]:           # tidak ada satu pun query yang berhasil
        result["error"] = state["error"]
        result["answer"] = "Sorry, I could not build a valid database query for this question."
        return result

    # Jawaban mengandung aksara non-Latin: minta LLM menulis ulang sekali, lalu hapus sisa aksara yang masih ada
    if answer and FOREIGN_SCRIPT.search(answer):
        answer = _text(llm.invoke([("human", f"Rewrite this answer in plain {language} using only Latin letters. "
                                    f"Keep every number and id exactly as it is.\n\n{answer}")]))
        answer = FOREIGN_SCRIPT.sub("", answer)
    result["answer"] = answer or "Sorry, I could not answer this question."
    return result


# =============================================================================
# TOOLS untuk supervisor (LLM supervisor hanya membaca nama + docstring tool)
# =============================================================================
# Role dan id TIDAK dijadikan argumen tool, karena argumen tool diisi oleh LLM supervisor dan bisa dibelokkan
# lewat prompt (mis. "saya sebenarnya staf company"). Karena itu tool dibuat per sesi login (closure): role dan
# id dikunci saat tool dibuat, dan LLM hanya bisa mengisi pertanyaannya.

# Akun demo seller, dipakai kalau tidak ada user_id dari halaman login. Id dibaca dari .env (DEMO_SELLER_ID),
# dengan nilai default di bawah. Customer tidak punya akun demo karena customer selalu pembeli anonim.
DEMO_USER_IDS = {
    "seller": os.getenv("DEMO_SELLER_ID", "d91fb3b7d041e83b64a00a3edfb37e4f"),         # 535 order
}


def demo_user_id(role):
    """Kembalikan id akun demo untuk role ini, atau None kalau role itu tidak punya akun demo
    (company dan customer). Fungsi ini juga dipakai oleh rag_agent."""
    return DEMO_USER_IDS.get((role or "").strip().lower())


# Penampung hasil query untuk ditampilkan di UI sebagai tabel/grafik. Sebelum tiap request, UI mengisi list
# kosong di sini; ask_data_agent lalu menambahkan hasil query-nya ({"sql", "data"}) ke list itu.
# Dipakai ContextVar (bukan list global) karena supervisor di-cache dan dipakai bersama antar sesi;
# LangGraph menyalin context ini ke thread pekerjanya.
collected_results = contextvars.ContextVar("collected_results", default=None)


def make_data_tools(role, user_id=None):
    """Buat tool data untuk SATU sesi login.

    role dan user_id diambil dari halaman login, bukan dari LLM. Seller WAJIB punya user_id: tidak ada
    fallback ke akun demo, supaya sesi tanpa id tidak diam-diam melihat data toko lain. Customer selalu anonim.

    Mengembalikan list berisi satu tool untuk supervisor: [ask_data_agent].
    Raise ValueError kalau login tidak valid (role tidak dikenal, id seller kosong atau tidak ada di database),
    supaya aplikasi bisa menolak sejak awal.
    """
    scope = make_scope(role, user_id)

    @tool
    def ask_data_agent(question: str) -> str:
        """Ask the DATA specialist anything that needs numbers from the Olist database: revenue, counts, averages,
        rankings, top sellers/customers/categories, payment types and installments, customer states, repeat
        buyers, delivery times, monthly/yearly trends, a customer's spending. The data is automatically limited
        to what the logged-in user may see. It cannot read review TEXT (use the review specialist for that).

        Arguments:
            question: a complete, self-contained question. ALWAYS say the answer language, e.g.
                "Who are my top 5 product categories by revenue in 2018? Answer in Bahasa Indonesia."
        """
        try:
            result = ask_sql_agent(question, role=scope.role, user_id=scope.user_id, verbose=False)
        except Exception as error:          # error apa pun dikembalikan sebagai teks agar supervisor tidak crash
            return f"The data agent failed: {type(error).__name__}: {str(error)[:200]}"
        sink = collected_results.get()      # simpan hasil query untuk UI (kalau UI menyiapkan list-nya)
        if sink is not None:
            sink.extend(result["results"])
        return result["answer"]

    return [ask_data_agent]
