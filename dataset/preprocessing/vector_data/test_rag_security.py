"""
test_rag_security.py - Tes OFFLINE (tanpa OpenAI / Qdrant) untuk scope per role di RAG agent.

Menguji: filter seller dikunci ke seller_id sendiri, customer anonim boleh mencari per seller tapi tidak per
order (aturan C2 di agent/policy.py) dan tidak melihat order_id, dan tool supervisor dibuat per sesi login.
rag_core.search diganti dengan versi palsu yang mencatat filter yang dikirim.

Jalankan dari folder utama proyek:
    python -m dataset.preprocessing.vector_data.test_rag_security
"""
import inspect

from agent import rag_agent
from dataset.preprocessing.numeric_data.sql_tools import get_connection, make_scope

SELLER = rag_agent.demo_user_id("seller")

results = []


def check(name, condition, detail=""):
    condition = bool(condition)
    results.append(condition)
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


conn = get_connection()
OTHER_SELLER = conn.execute("SELECT seller_id FROM sellers WHERE seller_id != ? LIMIT 1", (SELLER,)).fetchone()[0]
OWN_ORDER, OTHER_ORDER = [row[0] for row in conn.execute("SELECT order_id FROM orders LIMIT 2")]
conn.close()

# rag_core.search palsu: catat filter, kembalikan dua review dari dua order berbeda
sent = []


def fake_retrieve(query, top_k, filters):
    sent.append(filters)
    return [{"content": "chegou rapido", "relevance_score": 0.9,
             "metadata": {"order_id": OWN_ORDER, "review_score": 5, "seller_city": "sao paulo",
                          "product_name": ["toys"], "price": [10.0]}},
            {"content": "produto quebrado", "relevance_score": 0.8,
             "metadata": {"order_id": OTHER_ORDER, "review_score": 1, "seller_city": "curitiba",
                          "product_name": ["toys"], "price": [20.0]}}]


rag_agent.retrieve = fake_retrieve


def search(scope, **args):
    sent.clear()
    tool = rag_agent.make_search_tool(scope)
    return tool.invoke({"query": "produto", "answer_language": "English", **args})


company = make_scope("company")
seller = make_scope("seller", SELLER)
customer = make_scope("customer")

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

# ---------------------------------------------------------------- 3. customer anonim: per kategori, bukan per seller/order
out = search(customer, product_category="toys", max_rating=2, max_price=100)
check("customer: filter kategori/rating/harga dipakai apa adanya",
      sent and sent[0] == {"product_name": "toys", "review_score": {"lte": 2}, "price": {"lte": 100}}, str(sent))
check("customer: tidak ada order_id di hasil", OWN_ORDER not in out and "order_id" not in out.split("ANSWER LANGUAGE")[0])
check("customer: pengingat jawaban tidak meminta order_id", "order_id" not in out.split("ANSWER LANGUAGE")[-1])
out = search(customer, seller_city="sao paulo")
check("customer: filter kota seller boleh", sent and sent[0] == {"seller_city": "sao paulo"})
out = search(customer, seller_id=OTHER_SELLER)
check("customer: filter seller_id boleh (review seller rekomendasi)",
      sent and sent[0] == {"seller_id": OTHER_SELLER} and OTHER_ORDER not in out, str(sent))
out = search(customer, order_id=OTHER_ORDER)
check("customer: order_id ditolak (C2)", not sent and out.startswith("NOT ALLOWED (rule C2)"), out[:60])
out = search(customer, order_ids=[OWN_ORDER, OTHER_ORDER])
check("customer: daftar order_ids ditolak (C2)", not sent and out.startswith("NOT ALLOWED (rule C2)"), out[:60])
out = search(company)
check("company: order_id tetap terlihat", OTHER_ORDER in out)

# ---------------------------------------------------------------- 4. tool supervisor per sesi login
tools = rag_agent.make_review_tools("seller", SELLER)
args = set(inspect.signature(tools[0].func).parameters)
check("ask_review_agent hanya punya argumen question", args == {"question"}, str(args))
captured = []
original = rag_agent.ask_rag_agent
rag_agent.ask_rag_agent = lambda question, **kw: captured.append(kw) or "ok"
tools[0].invoke({"question": "Why do my customers complain?"})
rag_agent.ask_rag_agent = original
check("tool meneruskan role + id dari login", captured and captured[0] == {"role": "seller", "user_id": SELLER})
try:
    rag_agent.make_review_tools("seller", "0" * 32)
    check("login dengan id tidak dikenal ditolak", False)
except ValueError:
    check("login dengan id tidak dikenal ditolak", True)
check("role note seller menyebut id", SELLER in rag_agent.make_role_note(seller))
check("role note customer: anonim", "anonymous" in rag_agent.make_role_note(customer))
check("company tanpa role note", rag_agent.make_role_note(company) == "")
check("make_review_tools customer tanpa id", len(rag_agent.make_review_tools("customer")) == 1)
try:
    rag_agent.make_review_tools("seller")
    check("login seller tanpa id ditolak (tanpa fallback demo)", False)
except ValueError:
    check("login seller tanpa id ditolak (tanpa fallback demo)", True)

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
