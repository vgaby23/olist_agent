"""
rag_core.py - the SEARCH side: ask a question, get the matching reviews back
============================================================================

ingest.py WRITES reviews into Qdrant (done once).
rag_core.py READS them (done every time someone asks a question). This is the "R" in RAG:
Retrieval.

Who uses this file:
    app.py            the API calls search() for every request
    check_upload.py   your own test after an upload (no API needed)

WHAT search() DOES, STEP BY STEP
    1. Take the question, e.g. "late delivery", and the filters, e.g. rating 1 or 2 only.
    2. Turn the filters into a Qdrant filter                       (build_filter, below)
    3. Turn the question into a vector with the embedding model    (done by LangChain)
    4. Ask Qdrant for the reviews whose vectors are CLOSEST to the question's vector,
       looking only at reviews that pass the filter
    5. Return them as a simple list: text + score + metadata

TRY IT WITHOUT THE API (the .env file must contain your Qdrant and OpenAI keys)
    from dotenv import load_dotenv; load_dotenv()
    import rag_core
    results = rag_core.search("entrega atrasada", top_k=3, filters={"review_score": {"lte": 2}})
    print(results[0]["content"])

WHAT search() RETURNS: a list with one dictionary per review, best match first
    {"content": "A entrega atrasou ...",
     "relevance_score": 0.71,                       <- closer to 1 = better match
     "metadata": {"order_id": "o2", "review_score": 1, "seller_city": "sao paulo", ...}}
"""
import os

from qdrant_client import models

from dataset.preprocessing.vector_data import text_vectordb as vectordb

# The most reviews one request may ask for. app.py uses it to reject something like top_k=100000.
MAX_TOP_K = int(os.environ.get("MAX_TOP_K", "50"))


# =============================================================================
# Part 1: FILTERS
# =============================================================================
# A filter lets us say "only look at reviews where ...". The caller sends a simple dictionary:
#
#   {"review_score": {"lte": 2}}                  rating 2 or lower     (a RANGE)
#   {"review_score": 5}                           rating exactly 5
#   {"product_name": "health_beauty"}             one of the products is in this category
#   {"product_name": ["health_beauty", "toys"]}   ... in ANY of these categories
#   {"seller_city": "Sao Paulo"}                  upper/lower case does not matter
#   {"review_score": {"lte": 2}, "seller_city": "sao paulo"}      several filters = ALL must match
#
# Range words:  gt = greater than,  gte = greater than or equal,  lt = less than,  lte = less than or equal
#
# Only the fields listed below are allowed, so nobody can query other fields by accident.
# "match" = compare with a value,  "range" = compare with numbers (gt, gte, lt, lte).
FILTER_FIELDS = {
    "order_id": "match",
    "seller_id": "match",
    "seller_city": "match",
    "product_id": "match",
    "product_name": "match",
    "review_score": "range",
    "price": "range",
}
RANGE_WORDS = {"gt", "gte", "lt", "lte"}


class FilterError(ValueError):
    """The caller sent a filter we cannot use (unknown field, wrong kind of value ...).
    app.py catches this and answers with HTTP 400 and a clear message."""


def is_number(value):
    """True for 5 or 2.5, False for "5" and for True/False (Python treats True as the number 1)."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def clean_value(field, value):
    """Prepare ONE filter value so it can be compared with what is stored in Qdrant."""
    if value is None or isinstance(value, (dict, list)):
        raise FilterError(f"Invalid value for '{field}': {value!r}")

    if field == "seller_city":
        # We saved cities in lower-case without extra spaces ("sao paulo"), so do the same here.
        return str(value).strip().lower()

    if field == "review_score":
        if isinstance(value, bool):
            raise FilterError("review_score must be an integer")
        try:
            return int(value)                # "5" -> 5
        except (TypeError, ValueError):
            raise FilterError(f"review_score must be an integer, got {value!r}")

    return str(value)


def make_range_condition(field, key, value):
    """{"lte": 2}  ->  a Qdrant condition meaning "this field is <= 2"."""
    if FILTER_FIELDS[field] != "range":
        raise FilterError(f"'{field}' does not support range filters")

    if len(value) == 0 or set(value) - RANGE_WORDS:        # empty, or contains a word we do not know
        raise FilterError(f"Range for '{field}' must use keys from {sorted(RANGE_WORDS)}")

    for number in value.values():
        if not is_number(number):
            raise FilterError(f"Range values for '{field}' must be numbers")

    # **value unpacks {"lte": 2} into lte=2, so this is the same as models.Range(lte=2)
    return models.FieldCondition(key=key, range=models.Range(**value))


def make_any_of_condition(field, key, values):
    """["a", "b"]  ->  a condition meaning "the value is a OR b"."""
    if len(values) == 0:
        raise FilterError(f"'{field}' list must not be empty")

    cleaned = []
    for value in values:
        cleaned.append(clean_value(field, value))

    return models.FieldCondition(key=key, match=models.MatchAny(any=cleaned))


def make_exact_condition(field, key, value):
    """"health_beauty"  ->  a condition meaning "the value is exactly this".
    If the stored field is a LIST (like product_name), it matches when ANY item in the list equals it."""
    return models.FieldCondition(key=key, match=models.MatchValue(value=clean_value(field, value)))


def build_filter(filters):
    """Turn the caller's dictionary into a Qdrant filter. Returns None when there is nothing to filter."""
    if not filters:
        return None

    conditions = []
    for field, value in filters.items():
        if field not in FILTER_FIELDS:
            raise FilterError(f"Unknown filter '{field}'. Allowed: {sorted(FILTER_FIELDS)}")

        if value is None:                       # {"seller_city": None} means "no filter on this"
            continue

        # Our metadata is stored under "metadata", so "review_score" is "metadata.review_score" in Qdrant.
        key = vectordb.META + field

        if isinstance(value, dict):
            condition = make_range_condition(field, key, value)
        elif isinstance(value, list):
            condition = make_any_of_condition(field, key, value)
        else:
            condition = make_exact_condition(field, key, value)

        conditions.append(condition)

    if len(conditions) == 0:
        return None
    return models.Filter(must=conditions)      # "must" = ALL conditions have to be true


# =============================================================================
# Part 2: SEARCH
# =============================================================================
def search(query, top_k=5, filters=None):
    """Find the reviews that match the meaning of `query`.

    query    the question, e.g. "late delivery complaints"
    top_k    how many reviews to return
    filters  optional dictionary, see Part 1
    """
    qdrant_filter = build_filter(filters)      # raises FilterError if the filters are wrong

    # The "store" is LangChain's wrapper around our Qdrant collection (see text_vectordb.py).
    # similarity_search_with_score does steps 3 and 4 from the top of this file:
    # it embeds the query, then returns the closest reviews together with their scores.
    store = vectordb.get_store()
    hits = store.similarity_search_with_score(query, k=top_k, filter=qdrant_filter)

    results = []
    for document, score in hits:
        # LangChain adds internal keys that start with "_" (like "_id"). Remove them, so that
        # "metadata" contains only the fields we agreed on.
        metadata = {}
        for key, value in document.metadata.items():
            if not key.startswith("_"):
                metadata[key] = value

        results.append({
            "content": document.page_content,       # the review text
            "relevance_score": float(score),        # cosine similarity: closer to 1 = better match
            "metadata": metadata,
        })

    return results