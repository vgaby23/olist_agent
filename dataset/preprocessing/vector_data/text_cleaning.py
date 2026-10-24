"""
text_cleaning.py - STEP 1: read the CSV files and clean them
============================================================

Each function here reads ONE kind of CSV file and returns clean, simple Python data
(lists and dictionaries). Nothing in this file talks to Qdrant or to the embedding model.

    load_reviews         the review comments (cleaned; empty, short and duplicate ones removed)
    load_product_names   product_id -> English category name
    load_items           the items of every order (price rounded to a whole number)
    load_seller_cities   seller_id -> city (lower-case)

The next step (text_combinedata.py) joins these pieces together.

CSV FILES NEEDED in the data folder
  olist_order_reviews_dataset.csv         the comments we want to search
  olist_order_items_dataset.csv           which products / seller / price are in each order
  olist_products_dataset.csv              the category of each product (in Portuguese)
  product_category_name_translation.csv   Portuguese category -> English category
  olist_sellers_dataset.csv               the city of each seller
"""
import os
import re

import pandas as pd

# Comments shorter than this (in words) are skipped: "ok" or "bom" has no meaning to search.
MIN_WORDS = 3


def clean_text(text):
    """Tidy a text: one line, single spaces. An empty CSV cell becomes ''."""
    if pd.isna(text):                    # pandas reads an empty CSV cell as NaN ("missing")
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)     # any run of spaces / tabs / line breaks -> ONE space
    return text.strip()                  # remove spaces at the start and the end


def count_words(text):
    """Number of real words. A piece with no letter or digit in it (like "," or "!") is not a word,
    so ", muito ruim !" counts as 2 words, not 4."""
    count = 0
    for piece in text.split():
        if any(character.isalnum() for character in piece):
            count += 1
    return count


def load_reviews(data_dir, min_words=MIN_WORDS):
    """Return a list like [{"order_id": "o1", "review_score": 5, "comment": "..."}, ...]
    containing only reviews that have a real comment."""
    table = pd.read_csv(os.path.join(data_dir, "olist_order_reviews_dataset.csv"))
    print(f"  reviews in the file                : {len(table):,}")

    reviews = []
    already_kept = set()                       # (order_id, comment) pairs we already have

    for row in table.to_dict("records"):       # "records" = one dictionary per CSV row
        comment = clean_text(row["review_comment_message"])

        # Skip empty or very short comments.
        number_of_words = count_words(comment)
        if number_of_words < min_words:
            continue

        # Skip exact duplicates (same order, same text).
        key = (row["order_id"], comment)
        if key in already_kept:
            continue
        already_kept.add(key)

        reviews.append({
            "order_id": row["order_id"],
            "review_score": int(row["review_score"]),
            "comment": comment,
        })

    print(f"  reviews with a real comment        : {len(reviews):,}")
    return reviews


def load_product_names(data_dir):
    """Return a dictionary: {"p1": "health_beauty", "p2": "computers_accessories", ...}
    (product_id -> English category name).

    Olist has no product names, only categories. We call the field "product_name"
    because that is the name in the agreed API."""
    products = pd.read_csv(os.path.join(data_dir, "olist_products_dataset.csv"))
    translation = pd.read_csv(os.path.join(data_dir, "product_category_name_translation.csv"))

    # Build a lookup: {"beleza_saude": "health_beauty", ...}
    to_english = {}
    for row in translation.to_dict("records"):
        to_english[row["product_category_name"]] = row["product_category_name_english"]

    product_names = {}
    for row in products.to_dict("records"):
        portuguese = row["product_category_name"]

        if pd.isna(portuguese):
            name = "unknown"                           # this product has no category at all
        else:
            english = to_english.get(portuguese)       # None if there is no translation
            if english is None or pd.isna(english):
                name = portuguese                      # no translation: keep the Portuguese name
            else:
                name = english

        product_names[row["product_id"]] = name

    return product_names


def load_items(data_dir):
    """Return one dictionary per ordered item:
         [{"order_id": "o1", "product_id": "p1", "seller_id": "s1", "price": 30}, ...]
    Sorted so that item 1 of an order comes before item 2 of the same order."""
    table = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
    table = table.sort_values(["order_id", "order_item_id"])

    items = []
    for row in table.to_dict("records"):
        price = row["price"]
        if pd.isna(price):
            price = 0
        items.append({
            "order_id": row["order_id"],
            "product_id": row["product_id"],
            "seller_id": row["seller_id"],
            "price": int(round(price)),        # 29.9 -> 30   (cents are dropped)
        })
    return items


def load_seller_cities(data_dir):
    """Return {"s1": "sao paulo", "s2": "rio de janeiro", ...}.
    Cities are lower-case, so that a filter "Sao Paulo" or "sao paulo" both work."""
    sellers = pd.read_csv(os.path.join(data_dir, "olist_sellers_dataset.csv"))

    cities = {}
    for row in sellers.to_dict("records"):
        city = clean_text(row["seller_city"]).lower()
        if city == "":
            city = "unknown"
        if row["seller_id"] not in cities:     # if a seller appears twice, keep the first
            cities[row["seller_id"]] = city
    return cities