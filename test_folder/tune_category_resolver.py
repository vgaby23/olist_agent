"""
tune_category_resolver.py - check (and tune) agent/category_resolver.py on product words in 3 languages

For every word it prints the top 3 categories with their scores and whether the best one is acceptable.
Words in NONE should match no category at all. Use the printed scores to pick the thresholds in
agent/category_resolver.py (MATCH_MIN, STRONG_MATCH, MIN_GAP, CLOSE_MARGIN; each can be overridden by env var).
Needs OPENAI_API_KEY (embeddings only, costs a fraction of a cent; the categories are cached after the first run).

    python -m test_folder.tune_category_resolver
"""
from agent.category_resolver import CLOSE_MARGIN, MATCH_MIN, MIN_GAP, STRONG_MATCH, decide, resolve_category

# word -> categories that are an acceptable answer
CASES = {
    "make up": {"health_beauty"},
    "lipstick": {"health_beauty"},
    "lipstik": {"health_beauty"},
    "batom": {"health_beauty"},
    "skincare": {"health_beauty"},
    "Maybelline": {"health_beauty"},
    "shampoo": {"health_beauty"},
    "parfum": {"perfumery"},
    "running shoes": {"fashion_shoes"},
    "sepatu lari": {"fashion_shoes"},
    "tenis de corrida": {"fashion_shoes", "sports_leisure"},
    "kulkas": {"home_appliances"},
    "geladeira": {"home_appliances"},
    "blender": {"small_appliances"},
    "air fryer": {"small_appliances"},
    "HP Samsung": {"telephony"},
    "smartphone case": {"telephony"},
    "laptop": {"computers", "computers_accessories"},
    "mouse gaming": {"computers_accessories", "pc_gamer"},
    "jam tangan": {"watches_gifts"},
    "mainan anak": {"toys"},
    "boneka": {"toys"},
    "sprei": {"bed_bath_table"},
    "sofa": {"furniture_living_room"},
    "kasur": {"furniture_mattress_and_upholstery", "furniture_bedroom"},
    "makanan kucing": {"pet_shop"},
    "racao para cachorro": {"pet_shop"},
    "stroller bayi": {"baby"},
    "gitar": {"musical_instruments"},
    "bor listrik": {"costruction_tools_tools"},
    "health beauty": {"health_beauty"},
}
NONE = ["bitcoin", "concert tickets", "hotel booking", "car rental", "pizza delivery"]


def main():
    good = 0
    print(f"MATCH_MIN={MATCH_MIN} STRONG_MATCH={STRONG_MATCH} MIN_GAP={MIN_GAP} CLOSE_MARGIN={CLOSE_MARGIN}\n")
    for word, accepted in CASES.items():
        matches = resolve_category(word)
        decision = decide(matches)
        ok = decision["status"] != "none" and decision["categories"][0] in accepted
        good += ok
        print(f"[{'OK ' if ok else 'BAD'}] {word!r:24} {decision['status']:9} {matches}")
    for word in NONE:
        matches = resolve_category(word)
        decision = decide(matches)
        ok = decision["status"] == "none"
        good += ok
        print(f"[{'OK ' if ok else 'BAD'}] {word!r:24} {decision['status']:9} {matches}  (should be none)")
    total = len(CASES) + len(NONE)
    print(f"\n{good}/{total} correct")


if __name__ == "__main__":
    main()
