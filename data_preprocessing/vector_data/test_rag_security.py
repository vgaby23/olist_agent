"""
test_rag_security.py - Tes OFFLINE (tanpa OpenAI / Qdrant) untuk scope per role di RAG agent.

Menguji: filter seller dikunci ke seller_id sendiri, filter customer dikunci ke order_id miliknya sendiri
(filter seller_id dan order orang lain ditolak), dan tool supervisor dibuat per sesi login.
rag_core.search diganti dengan versi palsu yang mencatat filter yang dikirim.

Jalankan dari folder utama proyek:
    python -m data_preprocessing.vector_data.test_rag_security
"""
import inspect

from agent import RAG_agent
from data_preprocessing.numeric_data.sql_tools import get_connection, make_scope

SELLER = RAG_agent.demo_user_id("seller")
CUSTOMER = RAG_agent.demo_user_id("customer")

results = []


def check(name, condition, detail=""):
    condition = bool(condition)
    results.append(condition)
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


conn = get_connection()
OTHER_SELLER = conn.execute("SELECT seller_id FROM sellers WHERE seller_id != ? LIMIT 1", (SELLER,)).fetchone()[0]
OWN_ORDER = conn.execute("SELECT o.order_id FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
                         "WHERE c.customer_unique_id = ? LIMIT 1", (CUSTOMER,)).fetchone()[0]
OTHER_ORDER = conn.execute("SELECT o.order_id FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
                           "WHERE c.customer_unique_id != ? LIMIT 1", (CUSTOMER,)).fetchone()[0]
conn.close()

# rag_core.search palsu: catat filter, kembalikan satu review milik customer + satu milik orang lain
sent = []


def fake_retrieve(query, top_k, filters):
    sent.append(filters)
    return [{"content": "chegou rapido", "relevance_score": 0.9,
             "metadata": {"order_id": OWN_ORDER, "review_score": 5, "seller_city": "sao paulo",
                          "product_name": ["toys"], "price": [10.0]}},
            {"content": "produto quebrado", "relevance_score": 0.8,
             "metadata": {"order_id": OTHER_ORDER, "review_score": 1, "seller_city": "curitiba",
                          "product_name": ["toys"], "price": [20.0]}}]


RAG_agent.retrieve = fake_retrieve


def search(scope, **args):
    sent.clear()
    tool = RAG_agent.make_search_tool(scope)
    return tool.invoke({"query": "produto", "answer_language": "English", **args})


company = make_scope("company")
seller = make_scope("seller", SELLER)
customer = make_scope("customer", CUSTOMER)

# ---------------------------------------------------------------- 1. company: tidak dibatasi
out = search(company, seller_id=OTHER_SELLER)
check("company boleh filter seller_id mana saja", sent and sent[0].get("seller_id") == OTHER_SELLER)
check("company melihat semua order_id", OTHER_ORDER in out)

# ---------------------------------------------------------------- 2. seller: dikunci ke toko sendiri
out = search(seller)
check("seller tanpa filter -> seller_id dipaksa", sent and sent[0].get("seller_id") == SELLER)
out = search(seller, seller_id=SELLER.upper())
check("seller dengan id sendiri (huruf besar) boleh", sent and sent[0].get("seller_id") == SELLER)
out = search(seller, seller_id=OTHER_SELLER)
check("seller lain ditolak", not sent and out.startswith("NOT ALLOWED"), out[:60])

# ---------------------------------------------------------------- 3. customer: hanya order sendiri
own_orders = RAG_agent.own_order_ids(customer)
out = search(customer, product_category="toys")
check("customer tanpa filter order -> order_id dipaksa ke order sendiri",
      sent and isinstance(sent[0].get("order_id"), list) and set(sent[0]["order_id"]) == own_orders,
      str(sent[0].get("order_id")) if sent else "")
check("filter kategori customer tetap dipakai", sent and sent[0].get("product_name") == "toys")
check("order lain tidak ada di filter customer", sent and OTHER_ORDER not in sent[0]["order_id"])
out = search(customer, seller_id=OTHER_SELLER)
check("customer filter seller_id ditolak", not sent and out.startswith("NOT ALLOWED"), out[:60])
out = search(customer, order_id=OTHER_ORDER)
check("customer order orang lain ditolak", not sent and out.startswith("NOT ALLOWED"), out[:60])
out = search(customer, order_ids=[OWN_ORDER, OTHER_ORDER])
check("customer daftar order campuran ditolak", not sent and out.startswith("NOT ALLOWED"), out[:60])
out = search(customer, order_id=OWN_ORDER)
check("customer order sendiri boleh", sent and sent[0].get("order_id") == OWN_ORDER)

# ---------------------------------------------------------------- 4. tool supervisor per sesi login
tools = RAG_agent.make_review_tools("seller")
args = set(inspect.signature(tools[0].func).parameters)
check("ask_review_agent hanya punya argumen question", args == {"question"}, str(args))
captured = []
original = RAG_agent.ask_rag_agent
RAG_agent.ask_rag_agent = lambda question, **kw: captured.append(kw) or "ok"
tools[0].invoke({"question": "Why do my customers complain?"})
RAG_agent.ask_rag_agent = original
check("tool meneruskan role + id dari login", captured and captured[0] == {"role": "seller", "user_id": SELLER})
try:
    RAG_agent.make_review_tools("seller", "0" * 32)
    check("login dengan id tidak dikenal ditolak", False)
except ValueError:
    check("login dengan id tidak dikenal ditolak", True)
check("role note seller menyebut id", SELLER in RAG_agent.make_role_note(seller))
check("company tanpa role note", RAG_agent.make_role_note(company) == "")

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
