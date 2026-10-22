# run from the project root: python -m test_folder.tune_thresholds
#
# Helps choose MIN_MATCH and WEAK_MATCH (agent/rag_agent.py) on the FULL Qdrant index.
# It runs 20 test searches WITHOUT any score cut-off and prints the scores, so we can see
# where real topics end and nonsense begins. Only the search is called (one embedding per
# question), not the LLM, so it costs almost nothing.
#
# Output:
#   1. one line per question: best, 5th and 10th score + the start of the top 3 reviews
#   2. a summary per group
#   3. data/tune_thresholds.csv with every hit, to read the reviews in full (data/ is not in git)

# PART 1. Settings --------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()                      # must run before importing rag_core (it reads .env when imported)

import csv
import os

from dataset.preprocessing.vector_data import rag_core

TOP_K = 10
CSV_PATH = "data/tune_thresholds.csv"

SELLER_ID = "7c67e1448b00f6e969d365cea6b010ab"     # seller with the most 1-2 star reviews
PRODUCT_ID = "aca2eb7d00ea1a7b8ebd4e68314663af"    # best-selling product

# PART 2. The 20 test questions ---------------------------------------------------------------
# (id, search query, filters)
# A = real topic, B = filter-only, C = near-miss, D = nonsense
TESTS = [
    ("A1", "atraso na entrega", {"review_score": {"lte": 2}}),
    ("A2", "produto chegou quebrado", {"review_score": {"lte": 2}}),
    ("A3", "produto errado diferente do pedido", {}),
    ("A4", "entrega rápida", {"review_score": {"gte": 4}}),
    ("A5", "qualidade ruim do produto", {"review_score": {"lte": 2}}),
    ("A6", "faltou item no pedido", {}),
    ("A7", "elogio", {"product_name": "bed_bath_table", "review_score": {"gte": 4}}),
    ("A8", "reclamação", {"seller_city": "sao paulo", "review_score": {"lte": 2}}),
    ("A9", "produto quebrou", {"price": {"gte": 150}}),
    ("A10", "reembolso devolução do dinheiro", {}),
    ("B1", "reclamação", {"seller_id": SELLER_ID, "review_score": {"lte": 2}}),
    ("B2", "opinião sobre o produto", {"product_id": PRODUCT_ID}),
    ("B3", "reclamação", {"product_name": "health_beauty", "review_score": 1}),
    ("B4", "elogio", {"product_name": "watches_gifts", "review_score": {"gte": 4}}),
    ("C1", "pagamento com bitcoin", {}),
    ("C2", "aplicativo travando no celular", {}),
    ("C3", "envio internacional da China", {}),
    ("D1", "spaceship engines", {}),
    ("D2", "receita de bolo de chocolate", {}),
    ("D3", "quem ganhou a copa do mundo", {}),
]


# PART 3. Run the searches ----------------------------------------------------------------------
def short(text, size=45):
    """First characters of a review, on one line."""
    text = " ".join(text.split())
    return text[:size] + ("..." if len(text) > size else "")


def run_tests():
    """Search every test question and return {id: list of scores}. Also writes the CSV."""
    all_scores = {}
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["test_id", "query", "rank", "score", "rating", "category", "review"])

        print(f"{'id':4} {'query':36} {'best':>5} {'5th':>5} {'10th':>5}   top 3 reviews")
        print("-" * 130)
        for test_id, query, filters in TESTS:
            hits = rag_core.search(query, top_k=TOP_K, filters=filters)
            scores = [hit["relevance_score"] for hit in hits]
            all_scores[test_id] = scores

            for rank, hit in enumerate(hits, start=1):
                meta = hit["metadata"]
                writer.writerow([test_id, query, rank, round(hit["relevance_score"], 3),
                                 meta.get("review_score"), "|".join(meta.get("product_name", [])),
                                 hit["content"]])

            def at(n):                                   # the n-th score, or "-" if fewer hits
                return f"{scores[n - 1]:.2f}" if len(scores) >= n else "  -"
            top3 = " / ".join(short(hit["content"]) for hit in hits[:3])
            print(f"{test_id:4} {query[:36]:36} {at(1):>5} {at(5):>5} {at(10):>5}   {top3}")

    return all_scores


# PART 4. Summary per group ----------------------------------------------------------------------
def print_summary(all_scores):
    names = {"A": "real topics", "B": "filter-only", "C": "near-miss", "D": "nonsense"}
    print("\nSUMMARY (scores of the top 10 per question)")
    print(f"{'group':16} {'lowest best':>12} {'highest best':>13} {'lowest 10th':>12}")
    for letter, name in names.items():
        groups = [scores for test_id, scores in all_scores.items() if test_id.startswith(letter) and scores]
        best = [scores[0] for scores in groups]
        tenth = [scores[-1] for scores in groups]
        print(f"{letter} {name:14} {min(best):12.2f} {max(best):13.2f} {min(tenth):12.2f}")

    print("\nHow to read it:")
    print("  MIN_MATCH should be ABOVE the highest best score of group D (and ideally C),")
    print("  and BELOW the scores of the relevant reviews in groups A and B.")
    print(f"  Read the reviews in {CSV_PATH} to judge which ones are really relevant.")


if __name__ == "__main__":
    print_summary(run_tests())
