"""
category_resolver.py - turn a product TYPE or NAME into the closest Olist product CATEGORY
==========================================================================================

WHY
    The Olist data has no product names, only 73 categories with English names such as "health_beauty".
    Users ask about "make up", "lipstik", "sepatu lari" or "Maybelline". Neither the SQL agent nor the review
    agent can filter on those words, so the supervisor first asks this file for the closest category.

HOW (semantic search over the categories)
    1. utils/category_descriptions.json gives every category a short description: English name, Portuguese
       name and example product types in English, Bahasa Indonesia and Portuguese.
    2. Every PHRASE of a description (the names and each example product type) is embedded once (same
       embedding model as the review search) and cached in utils/.category_embeddings.json. The cache is
       rebuilt automatically when a description changes.
    3. The user's wording is embedded and compared with every phrase (cosine similarity). A category scores
       as high as its best phrase: short words ("kulkas") match a short phrase much better than one long text.
    4. decide() turns the scores into "match", "ambiguous" (two close candidates) or "none" (no category is
       close enough; never guess). The thresholds are tuned with test_folder/tune_category_resolver.py.

TRY IT
    python -m agent.category_resolver "make up" "sepatu lari" "kulkas"
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS_FILE = ROOT / "utils" / "category_descriptions.json"
CACHE_FILE = ROOT / "utils" / ".category_embeddings.json"

# Tuned with test_folder/tune_category_resolver.py (text-embedding-3-small, phrase scores):
#   product words score 0.63-1.0 (brands like "Maybelline" ~0.47 but far ahead of the runner-up),
#   non-products ("bitcoin", "hotel booking") score 0.37-0.47 with the runner-up right behind.
# So: below MATCH_MIN nothing matches; between MATCH_MIN and STRONG_MATCH the best must lead by MIN_GAP;
# a runner-up within CLOSE_MARGIN of a strong best is "too close to call".
MATCH_MIN = float(os.environ.get("CATEGORY_MATCH_MIN", "0.42"))
STRONG_MATCH = float(os.environ.get("CATEGORY_STRONG_MATCH", "0.60"))
MIN_GAP = float(os.environ.get("CATEGORY_MIN_GAP", "0.10"))
CLOSE_MARGIN = float(os.environ.get("CATEGORY_CLOSE_MARGIN", "0.03"))

# The review index (Qdrant, categories.json) and the SQL database spell ONE category differently.
SQL_NAMES = {"portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_and_food_preparers"}


def load_descriptions():
    """{category: description} from utils/category_descriptions.json."""
    return json.loads(DESCRIPTIONS_FILE.read_text(encoding="utf-8"))


def _normalize(text):
    """'Health Beauty' / 'health-beauty' -> 'health_beauty' (to catch exact category names without embeddings)."""
    return "_".join(text.strip().lower().replace("-", " ").replace("_", " ").split())


def _embedder():
    from dataset.preprocessing.vector_data.text_embedding import EMBEDDING_MODEL, get_embeddings
    return EMBEDDING_MODEL, get_embeddings()


def phrases(description):
    """'health and beauty (beleza_saude): makeup, lipstick; batom' ->
    ['health and beauty', 'beleza saude', 'makeup', 'lipstick', 'batom']"""
    head, _, examples = description.partition(":")
    english, _, portuguese = head.partition("(")
    found = [english, portuguese.split(")")[0].replace("_", " ").replace(" / ", ", ")]
    found += examples.replace(";", ",").split(",")
    return list(dict.fromkeys(part.strip() for part in found if part.strip()))


_cache = {}      # {"key": ..., "owners": [...], "matrix": np.ndarray}, filled on first use


def _phrase_vectors():
    """(owners, matrix): one unit vector per phrase, owners[i] = the category of phrase i.
    Read from the cache file, or embedded once and written to it."""
    descriptions = load_descriptions()
    model, embeddings = _embedder()
    # "phrases-v1": bump it when the cache format or phrases() changes, so old caches are rebuilt
    key = hashlib.sha256(("phrases-v1" + model + json.dumps(descriptions, sort_keys=True)).encode()).hexdigest()
    if _cache.get("key") == key:
        return _cache["owners"], _cache["matrix"]

    pairs = [(name, phrase) for name in sorted(descriptions) for phrase in phrases(descriptions[name])]
    stored = None
    if CACHE_FILE.is_file():
        try:
            stored = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stored = None
    if stored and stored.get("key") == key:
        vectors = stored["vectors"]
    else:
        vectors = embeddings.embed_documents([phrase for _, phrase in pairs])
        try:
            CACHE_FILE.write_text(json.dumps({"key": key, "vectors": vectors}), encoding="utf-8")
        except OSError:
            pass                                  # read-only folder: just keep the vectors in memory
    matrix = np.array(vectors, dtype=float)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    owners = [name for name, _ in pairs]
    _cache.update(key=key, owners=owners, matrix=matrix)
    return owners, matrix


def resolve_category(text, top_k=3):
    """'make up' -> [("health_beauty", 0.52), ("perfumery", 0.41), ...] (best first).
    An exact category name (English or Portuguese, any spacing/case) returns [(category, 1.0)]."""
    text = (text or "").strip()
    if not text:
        return []
    descriptions = load_descriptions()
    wanted = _normalize(text)
    for name, description in descriptions.items():
        portuguese = description.split("(", 1)[1].split(")", 1)[0].split(" / ")[0] if "(" in description else ""
        if wanted in (name, _normalize(portuguese), SQL_NAMES.get(name)):
            return [(name, 1.0)]

    owners, matrix = _phrase_vectors()
    _, embeddings = _embedder()
    query = np.array(embeddings.embed_query(text), dtype=float)
    scores = matrix @ (query / np.linalg.norm(query))
    best = {}                                     # category -> its best phrase score
    for owner, score in zip(owners, scores):
        best[owner] = max(best.get(owner, -1.0), float(score))
    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [(name, round(score, 3)) for name, score in ranked]


def decide(matches):
    """Scores -> {"status": "match" | "ambiguous" | "none", "categories": [...]} (categories best first)."""
    if not matches or matches[0][1] < MATCH_MIN:
        return {"status": "none", "categories": []}
    top = matches[0][1]
    second = matches[1][1] if len(matches) > 1 else 0.0
    if top < STRONG_MATCH and top - second < MIN_GAP:       # weak AND not distinctive: probably not a product
        return {"status": "none", "categories": []}
    close = [name for name, score in matches[1:] if score >= MATCH_MIN and top - score <= CLOSE_MARGIN]
    if close:
        return {"status": "ambiguous", "categories": [matches[0][0], *close]}
    return {"status": "match", "categories": [matches[0][0]]}


def _label(name):
    sql = SQL_NAMES.get(name)
    return f"{name} (in the SQL database: {sql})" if sql else name


# The supervisor's tool. Only the name + docstring are read by the LLM.
@tool
def find_product_category(product: str) -> str:
    """Find the Olist product CATEGORY for a product type, product name or brand the user mentioned
    (e.g. "make up", "lipstik", "running shoes", "kulkas", "Maybelline"). The data only knows categories,
    so call this BEFORE asking the SQL or review specialist about a product that is not an exact category name,
    and put the returned exact category name into the specialist request.

    Arguments:
        product: the user's wording for the product, as short as possible (e.g. "make up", not the whole question).
    """
    try:
        matches = resolve_category(product)
    except Exception as error:               # e.g. no internet: say it, never crash the supervisor
        return f"The category lookup failed ({type(error).__name__}). Ask the user which product category they mean."
    decision = decide(matches)
    scores = ", ".join(f"{name} {score:.2f}" for name, score in matches)
    if decision["status"] == "match":
        return (f"Closest category for \"{product}\": {_label(decision['categories'][0])}. Use this exact name. "
                f"(scores: {scores})")
    if decision["status"] == "ambiguous":
        names = " or ".join(_label(name) for name in decision["categories"])
        return (f"\"{product}\" fits several categories about equally: {names}. Use all of them if that answers the "
                f"question naturally, otherwise ask the user which one they mean. (scores: {scores})")
    return (f"No product category matches \"{product}\" closely enough (best: {scores or 'none'}). Tell the user "
            f"that Olist has no matching category; do not guess one.")


if __name__ == "__main__":
    for word in sys.argv[1:] or ["make up"]:
        found = resolve_category(word)
        print(f"{word!r}: {decide(found)['status']}  {found}")
