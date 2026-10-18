# run from the project root: python -m test_folder.count_drops
import os
import pandas as pd
from data_preprocessing.vector_data import text_cleaning as tc

DATA_DIR = "./Dataset"        # the folder with the Olist CSV files

table = pd.read_csv(os.path.join(DATA_DIR, "olist_order_reviews_dataset.csv"))
total = len(table)

empty = short = duplicate = kept = 0
seen = set()
for row in table.to_dict("records"):
    comment = tc.clean_text(row["review_comment_message"])
    if comment == "":
        empty += 1
    elif tc.count_words(comment) < tc.MIN_WORDS:
        short += 1
    elif (row["order_id"], comment) in seen:
        duplicate += 1
    else:
        seen.add((row["order_id"], comment))
        kept += 1

print(f"reviews in the file          : {total:,}")
print(f"empty (no comment)           : {empty:,}")
print(f"under {tc.MIN_WORDS} words                : {short:,}")
print(f"exact duplicates             : {duplicate:,}")
print(f"kept                         : {kept:,}   (should be 35,230)")