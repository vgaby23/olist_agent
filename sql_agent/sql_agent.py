"""
sql_agent.py - SQL Agent untuk Olist E-Commerce AI Assistant.

Alur kerja:
  pertanyaan -> LLM menulis SQL -> run_query (aman, read-only)
             -> kalau error: LLM memperbaiki SQL (maks. MAX_RETRIES kali)
             -> LLM merangkum hasil menjadi jawaban singkat.
"""
import os
import re

# Aksara non-Latin (Ibrani, Arab, Thai, Cyrillic, CJK, dst.) yang tidak boleh muncul di jawaban
FOREIGN_SCRIPT = re.compile(r"[\u0400-\u04FF\u0590-\u06FF\u0900-\u097F\u0E00-\u0E7F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]")

from sql_tools import get_schema, run_query

MAX_RETRIES = 2

RULES = """
You are an expert SQLite analyst for the Olist Brazilian e-commerce database.
Write ONE read-only SQLite query (SELECT or WITH ... SELECT) that answers the user's question.

DATABASE SCHEMA:
{schema}

BUSINESS RULES (follow them strictly):
1. Revenue / spending / sales / ranking by sales: only count delivered orders -> JOIN orders and filter
   orders.is_delivered = 1. Revenue = SUM(order_items.price). Include freight_value only if the user asks for it.
   The same delivered-only rule applies to counting buyers, repeat customers, payment-method usage and any other
   customer-behaviour statistic: always join orders and filter orders.is_delivered = 1, unless the question is
   explicitly about order status (for example canceled orders).
2. Time trends (monthly/yearly): you MUST filter orders.in_trend_period = 1 (do not write month ranges by hand)
   and group by orders.purchase_month ('YYYY-MM').
3. A customer is identified by customers.customer_unique_id (customer_id is unique per order, NOT per person).
   Join: orders.customer_id = customers.customer_id. When counting customers ALWAYS use
   COUNT(DISTINCT customers.customer_unique_id); never COUNT(*) or COUNT(customer_id) on the customers table.
   Repeat customers = customer_unique_id with COUNT(DISTINCT orders.order_id) > 1 among delivered orders.
4. Product category: use products.product_category_name_english (the value 'unknown' means missing category).
   Products have NO name, only product_id and category.
5. Reviews: order_reviews has exactly one row per order (scores only, no review text). Average score =
   AVG(order_reviews.review_score), joined through order_id.
6. On-time delivery: orders.is_late is 1 (late) or 0 (on time) for delivered orders, NULL otherwise.
   On-time rate = 1 - AVG(is_late) over orders with is_delivered = 1. Delivery time = orders.delivery_days.
7. Payments: order_payments can have several rows per order. NEVER SUM values after joining order_items
   and order_payments together (rows multiply). Aggregate each table separately, or use orders.order_total.
8. Sellers have NO name: identify them by seller_id plus seller_city and seller_state.
9. The database has NO cost, profit, or commission data. Product dimensions/weight are not available either.
10. Use LIMIT 10 for rankings unless the user asks for a different number. Round money to 2 decimals.
11. Only use tables and columns that exist in the schema.

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
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))


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


def ask_sql_agent(question, role="company", llm=None, verbose=True, language="Indonesian"):
    """
    Jawab pertanyaan berbasis data. Mengembalikan dict:
      {"question", "sql", "data" (DataFrame atau None), "answer", "error"}
    """
    llm = llm or get_llm()
    system_prompt = RULES.format(schema=get_schema())
    role_note = f"The user's role is: {role}."

    messages = [("system", system_prompt), ("human", f"{role_note}\nQuestion: {question}")]
    sql, data, error = None, None, None

    for attempt in range(MAX_RETRIES + 1):
        reply = _clean_sql(_text(llm.invoke(messages)))

        if reply.upper().startswith("CANNOT_ANSWER"):
            reason = reply.split(":", 1)[-1].strip()
            refusal_prompt = (
                f"Politely tell the user, in one or two short sentences written in {language}, "
                f"that you cannot answer it because: {reason}. Do not invent any data.\n\n"
                f"Question: {question}"
            )
            return {"question": question, "sql": None, "data": None, "error": None,
                    "answer": _text(llm.invoke([("human", refusal_prompt)]))}

        sql = reply
        if verbose:
            print(f"[Percobaan {attempt + 1}] SQL:\n{sql}\n")
        try:
            data = run_query(sql)
            error = None
            break
        except Exception as e:  # SQL salah / ditolak -> minta LLM memperbaiki
            error = str(e)
            data = None
            if verbose:
                print(f"   -> Error: {error}\n")
            messages += [("ai", reply),
                         ("human", f"That query failed with this error: {error}\n"
                                   "Fix it and return ONLY the corrected SQL query.")]

    if error:
        return {"question": question, "sql": sql, "data": None, "error": error,
                "answer": "Maaf, saya belum berhasil membuat query yang valid untuk pertanyaan ini."}

    # Ringkas hasil menjadi jawaban untuk pengguna
    table = data.to_string(index=False) if not data.empty else "(no rows returned)"
    summary_prompt = (
        "You are a helpful e-commerce data assistant. Answer the user's question using ONLY the query result below.\n"
        f"- Write the whole answer in {language} only. Do not mix in any other language and do not use any "
        "non-Latin writing system.\n"
        "- Be concise. Do not mention column names, table names or SQL.\n"
        "- All money values are in Brazilian Real; write them as R$ with Indonesian-style number format "
        "(dot for thousands, comma for decimals). Do not convert currencies.\n"
        "- Do not invent numbers. If the result is empty, say that no data was found.\n"
        "- Sellers are identified by seller_id only (no names).\n"
        "- Payment types: write credit_card as 'kartu kredit', debit_card as 'kartu debit', "
        "boleto as 'boleto (transfer bank Brasil)', voucher as 'voucher'.\n\n"
        f"Question: {question}\n\nQuery result:\n{table}"
    )
    answer = _text(llm.invoke([("human", summary_prompt)]))
    if FOREIGN_SCRIPT.search(answer):  # jaga-jaga: minta ulang sekali, lalu bersihkan kalau masih ada
        answer = _text(llm.invoke([("human", summary_prompt + f"\n\nYour previous answer contained foreign "
                                    f"characters. Rewrite it in plain {language} using only Latin letters.")]))
        answer = FOREIGN_SCRIPT.sub("", answer)
    return {"question": question, "sql": sql, "data": data, "error": None, "answer": answer}
