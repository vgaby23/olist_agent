"""
sql_agent.py - SQL Agent untuk Olist E-Commerce AI Assistant.

Alur kerja:
  pertanyaan -> cek role + id (make_scope) -> LLM menulis SQL
             -> run_query (read-only, dibatasi sesuai role lewat TEMP VIEW + authorizer)
             -> kalau error: LLM memperbaiki SQL (maks. MAX_RETRIES kali)
             -> LLM merangkum hasil menjadi jawaban singkat.

Catatan:
  - role ditegakkan di kode (sql_tools.make_scope / run_query(scope=...)), bukan hanya di prompt
  - periode default 2017-01..2018-08; data di luar periode itu parsial; 2018 hanya Januari-Agustus
  - label kategori kosong = 'uncategorized'; view category_stats untuk pembanding se-platform
"""
import os
import re

import pandas as pd
from langchain_core.tools import tool

from data_preprocessing.numeric_data.sql_tools import get_connection, get_schema, make_scope, run_query

try:                                     # agent/language.py dipakai bersama RAG agent
    from agent.language import guess_language
except ImportError:                      # sementara file itu belum ada: biarkan LLM mengikuti bahasa pertanyaan
    def guess_language(question):
        return None

# Aksara non-Latin (Ibrani, Arab, Thai, Cyrillic, CJK, dst.) yang tidak boleh muncul di jawaban
FOREIGN_SCRIPT = re.compile(r"[Ѐ-ӿ֐-ۿऀ-ॿ฀-๿぀-ヿ一-鿿가-힯]")

MAX_RETRIES = 2

# Hanya dipakai kalau jawabannya dalam Bahasa Indonesia
INDONESIAN_TERMS = ("- Payment types: write credit_card as 'kartu kredit', debit_card as 'kartu debit', "
                    "boleto as 'boleto (transfer bank Brasil)', voucher as 'voucher'.\n")

RULES = """
You are an expert SQLite analyst for the Olist Brazilian e-commerce database.
Write ONE read-only SQLite query (SELECT or WITH ... SELECT) that answers the user's question.

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
   Yearly: GROUP BY orders.purchase_year.
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
6. On-time delivery: orders.is_late is 1 (late) or 0 (on time) for delivered orders, NULL otherwise.
   On-time rate = 1 - AVG(is_late) over orders with is_delivered = 1. Delivery time = orders.delivery_days.
7. Payments: order_payments can have several rows per order. NEVER SUM values after joining order_items
   and order_payments together (rows multiply). Aggregate each table separately, or use orders.order_total.
   Installments: AVG(payment_installments), usually grouped by payment_type.
8. Sellers have NO name: identify them by seller_id plus seller_city and seller_state.
9. The database has NO cost, profit, or commission data. Product dimensions/weight are not available either.
10. Use LIMIT 10 for rankings and lists unless the user asks for a different number. Round money to 2 decimals.
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

OUTPUT FORMAT:
- Return ONLY the SQL query, no explanation, no markdown fences.
- If the question cannot be answered with this database (for example profit, product name, seller name),
  return exactly: CANNOT_ANSWER: <short reason in English>
""".strip()


def get_llm():
    """Buat koneksi ke OpenAI (key dan model dibaca dari file .env)."""
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()
    return ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)   # 0 = SQL yang konsisten


def _text(response):
    """Ambil teks dari respons LLM (bisa string, bisa daftar blok)."""
    content = response.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content.strip()


def _clean_sql(text):
    """Buang pagar ```sql ... ``` kalau LLM menambahkannya."""
    text = re.sub(r"^```(?:sql)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def pick_language(question, language=None):
    """Bahasa jawaban: 1) argumen language kalau diisi, 2) petunjuk "Answer in X" dari supervisor,
    3) kalau tidak ada: tebak dari kata-kata di pertanyaan (guess_language)."""
    if language:
        return language
    hint = re.search(r"answer in ([A-Za-z][A-Za-z ]{1,30}?)\s*[.!\n]", question + "\n", re.IGNORECASE)
    if hint:
        return hint.group(1).strip()
    return guess_language(question) or "the same language as the question"


def make_role_note(scope):
    """Penjelasan untuk LLM tentang siapa penggunanya dan data apa yang terlihat (scope sudah ditegakkan di kode)."""
    if scope.role == "seller":
        return (f"The user is the seller {scope.user_id}. The tables sellers, order_items, orders, order_payments, "
                "order_reviews and customers ALREADY contain only this seller's own data, so 'my', 'me', 'I', "
                "'saya', 'toko saya' simply mean all rows; no seller filter is needed. Other sellers' data is not "
                "available. In orders, items_value/freight_total/n_items/order_total count only this seller's items, "
                "and order_payments.payment_value is this seller's share of each payment (orders can also contain "
                "other sellers' items). For revenue still use SUM(order_items.price). For comparisons with the "
                "market use category_stats (platform-wide, per category).")
    if scope.role == "customer":
        return (f"The user is the customer {scope.user_id}. The tables customers, orders, order_items, "
                "order_payments, order_reviews and sellers ALREADY contain only this customer's own data, so "
                "'my', 'me', 'I', 'saya' simply mean all rows. Other customers' data is not available.")
    return "The user is a company (platform) employee and can see all data."


def period_note(sql):
    """Catatan periode untuk prompt ringkasan, sesuai filter yang BENAR-BENAR ada di SQL (bukan asumsi)."""
    lowered = sql.lower()
    if "in_trend_period" in lowered:
        note = ("- The numbers cover January 2017 to August 2018 (the default analysis period); say so briefly. "
                "If years are compared or a 2018 total is given, mention that 2018 only contains January to August.")
    elif re.search(r"purchase_month|purchase_year|order_purchase_timestamp", lowered):
        note = ("- The query filters on the period the user asked for. If that period includes months before "
                "2017-01 or after 2018-08, say the data for those months is partial. If years are compared or a "
                "2018 total is given, mention that 2018 only contains January to August.")
    else:
        note = ("- The query has no period filter, so the numbers cover all available data (September 2016 to "
                "October 2018). Do not claim any narrower period.")
    return note + "\n"


def _say(llm, language, message, question):
    """Kalimat singkat ke pengguna (penolakan / batasan) dalam bahasa jawaban."""
    prompt = (f"Politely tell the user, in one or two short sentences written in {language}, "
              f"that you cannot answer it because: {message}. Do not invent any data.\n\nQuestion: {question}")
    return _text(llm.invoke([("human", prompt)]))


def ask_sql_agent(question, role="company", llm=None, verbose=True, language=None, user_id=None):
    """
    Jawab pertanyaan berbasis data. Mengembalikan dict:
      {"question", "role", "sql", "data" (DataFrame atau None), "answer", "error"}
    user_id: seller_id (role "seller") atau customer_unique_id (role "customer") dari halaman login.
             Wajib untuk seller dan customer; tanpa id yang valid agent menolak menjawab.
    """
    llm = llm or get_llm()
    language = pick_language(question, language)
    result = {"question": question, "role": role, "sql": None, "data": None, "error": None}

    try:
        scope = make_scope(role, user_id)      # role ditegakkan di kode
    except ValueError as error:
        result["error"] = str(error)
        result["answer"] = _say(llm, language, f"the login is incomplete ({error})", question)
        return result

    messages = [("system", RULES.format(schema=get_schema())),
                ("human", f"{make_role_note(scope)}\nQuestion: {question}")]
    sql, data, error = None, None, None

    for attempt in range(MAX_RETRIES + 1):
        reply = _clean_sql(_text(llm.invoke(messages)))

        if reply.upper().startswith("CANNOT_ANSWER"):
            result["answer"] = _say(llm, language, reply.split(":", 1)[-1].strip(), question)
            return result

        sql = reply
        if verbose:
            print(f"[Percobaan {attempt + 1}] SQL:\n{sql}\n")
        try:
            data = run_query(sql, scope=scope)
            error = None
            break
        except Exception as e:  # SQL salah / ditolak -> minta LLM memperbaiki
            error = f"{type(e).__name__}: {e}"
            data = None
            if verbose:
                print(f"   -> Error: {error}\n")
            messages += [("ai", reply),
                         ("human", f"That query failed with this error: {error}\n"
                                   "Fix it and return ONLY the corrected SQL query.")]

    result["sql"] = sql
    if error:
        result["error"] = error
        result["answer"] = "Sorry, I could not build a valid database query for this question."
        return result

    # Ringkas hasil menjadi jawaban untuk pengguna
    table = data.to_string(index=False) if not data.empty else "(no rows returned)"
    if data.attrs.get("truncated"):
        table += f"\n(Only the first {len(data)} rows are shown; the full result has more. Tell the user.)"
    summary_prompt = (
        "You are a helpful e-commerce data assistant. Answer the user's question using ONLY the query result below.\n"
        f"- Write the whole answer in {language} only. Do not mix in any other language and do not use any "
        "non-Latin writing system.\n"
        "- Be concise. Do not mention column names, table names or SQL.\n"
        "- All money values are in Brazilian Real; write them as R$ with the usual number format of the answer "
        "language (Indonesian/Portuguese: 1.234,56; English: 1,234.56). Do not convert currencies.\n"
        "- Do not invent numbers. If the result is empty, say that no data was found.\n"
        "- Sellers are identified by seller_id only (no names).\n"
        "- Copy every id (seller_id, order_id, product_id, customer_unique_id) in FULL, exactly as in the result. "
        "Never shorten an id.\n"
        "- Copy city names, state names and category names EXACTLY as they appear in the result "
        "(same spelling; you may only capitalise them). Never replace them with a similar-looking name.\n"
        f"{period_note(sql)}"
        f"{INDONESIAN_TERMS if 'indones' in language.lower() else ''}\n"
        f"Question: {question}\n\nSQL used:\n{sql}\n\nQuery result:\n{table}"
    )
    answer = _text(llm.invoke([("human", summary_prompt)]))
    if FOREIGN_SCRIPT.search(answer):  # jaga-jaga: minta ulang sekali, lalu bersihkan kalau masih ada
        answer = _text(llm.invoke([("human", summary_prompt + f"\n\nYour previous answer contained foreign "
                                    f"characters. Rewrite it in plain {language} using only Latin letters.")]))
        answer = FOREIGN_SCRIPT.sub("", answer)
    result.update(sql=sql, data=data, answer=answer)
    return result


# =============================================================================
# TOOLS untuk supervisor (yang dibaca LLM supervisor hanyalah nama + docstring)
# =============================================================================
@tool
def ask_data_agent(question: str, role: str = "company", user_id: str = "") -> str:
    """Ask the DATA specialist anything that needs numbers from the Olist database: revenue, counts, averages,
    rankings, top sellers/customers/categories, payment types and installments, customer states, repeat buyers,
    delivery times, monthly/yearly trends, a customer's spending. It cannot read review TEXT (use the review
    specialist for what customers say).

    Arguments:
        question: a complete, self-contained question. ALWAYS say the answer language, e.g.
            "Who are the top 5 sellers by revenue in 2018? Answer in Bahasa Indonesia."
        role: the logged-in user's role: "customer", "seller" or "company".
        user_id: the logged-in seller's seller_id or customer's customer_unique_id (empty for company).
            Required for seller and customer; their data is automatically limited to that id.
    """
    try:
        result = ask_sql_agent(question, role=role, verbose=False, user_id=user_id or None)
    except Exception as error:          # never crash the supervisor
        return f"The data agent failed: {type(error).__name__}: {str(error)[:200]}"
    return result["answer"]


def get_customer_categories(customer_unique_id, limit=3):
    """Kategori yang paling sering dibeli seorang pelanggan (order terkirim saja). DataFrame, bisa kosong."""
    sql = """
        SELECT p.product_category_name_english AS category,
               COUNT(*)                         AS items_bought,
               MAX(o.order_purchase_timestamp)  AS last_bought
        FROM customers c
        JOIN orders o       ON o.customer_id = c.customer_id
        JOIN order_items oi ON oi.order_id = o.order_id
        JOIN products p     ON p.product_id = oi.product_id
        WHERE c.customer_unique_id = ?
          AND o.is_delivered = 1
          AND p.product_category_name_english != 'uncategorized'
        GROUP BY category
        ORDER BY items_bought DESC, last_bought DESC
        LIMIT ?
    """
    conn = get_connection()
    try:
        return pd.read_sql_query(sql, conn, params=(customer_unique_id.strip(), int(limit)))
    finally:
        conn.close()


@tool
def customer_past_categories(customer_unique_id: str) -> str:
    """Return the product categories a customer bought before (most bought first), for recommendations.
    Use the English category names it returns as product_category in the review specialist's search.

    Arguments:
        customer_unique_id: the customer's long unique id (customers.customer_unique_id).
    """
    try:
        data = get_customer_categories(customer_unique_id)
    except Exception as error:
        return f"Could not read the purchase history: {type(error).__name__}: {str(error)[:200]}"
    if data.empty:
        return f"No delivered orders found for customer {customer_unique_id}."
    lines = [f"- {row.category} ({row.items_bought} item(s), last bought {row.last_bought[:10]})"
             for row in data.itertuples()]
    return f"Customer {customer_unique_id} bought before:\n" + "\n".join(lines)
