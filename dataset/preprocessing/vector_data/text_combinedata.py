"""
text_combinedata.py - STEP 2: combine the cleaned CSV data into one Document per review
=======================================================================================

text_cleaning.py gives us separate pieces (reviews, items, product names, seller cities).
Here we JOIN them, so that every review comment gets the facts we want to filter on.

WHAT ONE DOCUMENT LOOKS LIKE
  Inputs:
     review   : order o1, score 5, comment "produto chegou antes do prazo"
     items    : order o1 has two items:  p1 (health_beauty, price 30)
                                         p2 (computers_accessories, price 100)
     seller   : s1 is in "sao paulo"
  Result:
     page_content = "produto chegou antes do prazo"        <- the text that gets embedded
     metadata     = {"order_id": "o1", "product_id": ["p1", "p2"], "seller_id": "s1",
                     "seller_city": "sao paulo", "product_name": ["health_beauty",
                     "computers_accessories"], "review_score": 5, "price": [30, 100]}
                                                            <- facts we can filter on
"""
import uuid

from langchain_core.documents import Document

from dataset.preprocessing.vector_data import text_cleaning


def group_items_by_order(items, product_names):
    """An order can have several items, but we index ONE document per review.
    So we collect the items of each order into lists.

    Returns: {"o1": {"product_id": ["p1", "p2"],
                     "product_name": ["health_beauty", "computers_accessories"],
                     "price": [30, 100],
                     "seller_id": "s1"},
              "o2": {...}, ...}

    The three lists have the same order: product_id[0], product_name[0] and price[0]
    all describe the first item."""
    orders = {}
    for item in items:
        order_id = item["order_id"]

        # First time we see this order: create its empty lists.
        if order_id not in orders:
            orders[order_id] = {
                "product_id": [],
                "product_name": [],
                "price": [],
                # An order can have several sellers; the API has ONE seller_id,
                # so we use the seller of the first item.
                "seller_id": item["seller_id"],
            }
        order = orders[order_id]

        order["product_id"].append(item["product_id"])
        order["product_name"].append(product_names.get(item["product_id"], "unknown"))
        order["price"].append(item["price"])

    return orders


def make_id(order_id, comment):
    """Every point in Qdrant needs an id. uuid5 turns any text into a UUID, and the same
    text always gives the SAME UUID. So running ingest twice overwrites the old
    points instead of creating duplicates."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, order_id + "|" + comment))


def build_documents(data_dir=".", min_words=text_cleaning.MIN_WORDS):
    """Load the CSV files (text_cleaning.py), combine them, and return three things:
         documents  : list of Document(page_content=comment, metadata={...})
         ids        : list of ids, one per document
         categories : sorted list of all category names (saved for the agent)"""
    print("Reading the CSV files ...")
    reviews = text_cleaning.load_reviews(data_dir, min_words)
    product_names = text_cleaning.load_product_names(data_dir)
    items = text_cleaning.load_items(data_dir)
    seller_cities = text_cleaning.load_seller_cities(data_dir)

    orders = group_items_by_order(items, product_names)

    documents = []
    ids = []
    skipped_no_items = 0

    for review in reviews:
        order = orders.get(review["order_id"])     # the items of this review's order

        # A few orders have no items in the items file. We cannot fill the product
        # fields for them, so we skip them.
        if order is None:
            skipped_no_items += 1
            continue

        seller_id = order["seller_id"]
        metadata = {
            "order_id": review["order_id"],
            "product_id": order["product_id"],
            "seller_id": seller_id,
            "seller_city": seller_cities.get(seller_id, "unknown"),
            "product_name": order["product_name"],
            "review_score": review["review_score"],
            "price": order["price"],
        }

        documents.append(Document(page_content=review["comment"], metadata=metadata))
        ids.append(make_id(review["order_id"], review["comment"]))

    print(f"  skipped (order has no items)       : {skipped_no_items:,}")
    print(f"  documents ready to index           : {len(documents):,}")

    categories = sorted(set(product_names.values()) - {"unknown"})
    return documents, ids, categories