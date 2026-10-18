"""
ingest.py - put the Olist review comments into Qdrant (the vector database)
==========================================================================

This file only RUNS the steps in order. The work happens in four files:

    text_cleaning.py      1. read the CSV files and clean them
    text_combinedata.py   2. combine them: one Document per review (comment + facts)
    text_embedding.py     3. the embedding model (text -> vector)
    text_vectordb.py      4. Qdrant: create the collection, embed and upload

Run it once, and again only when the data changes. The API (app.py) later searches
what this script uploaded.

HOW TO RUN
  # 1) Try it WITHOUT Qdrant first: builds the documents and prints 3 examples
  python ingest.py --data-dir ./data --dry-run

  # 2) The real thing (needs Qdrant details and the embedding model)
  export QDRANT_URL=https://xxxx.cloud.qdrant.io:6333
  export QDRANT_API_KEY=your-key
  python ingest.py --data-dir ./data --limit 500     # small test: only 500 reviews
  python ingest.py --data-dir ./data                 # everything
"""
import argparse
import json
import os

from dotenv import load_dotenv

# Load the variables from the .env file (QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY ...).
# This MUST run before the two imports below: they read these variables the moment they are imported.
load_dotenv()

from data_preprocessing.vector_data import text_combinedata
from data_preprocessing.vector_data import text_vectordb

# The agent's image step needs the list of valid category names. We save it here.
CATEGORIES_FILE = os.environ.get("CATEGORIES_FILE", "categories.json")


def save_categories(categories):
    with open(CATEGORIES_FILE, "w", encoding="utf-8") as file:
        json.dump(categories, file, ensure_ascii=False, indent=2)


def ingest(data_dir=".", recreate=True, batch_size=64, limit=None):
    """The whole job: build documents -> create collection -> embed and upload."""
    # Steps 1 + 2: read, clean and combine the CSV files
    documents, ids, categories = text_combinedata.build_documents(data_dir)

    if limit:                                   # quick test: only the first N documents
        documents = documents[:limit]
        ids = ids[:limit]

    # Steps 3 + 4: create the collection, then embed and upload
    text_vectordb.ensure_collection(recreate)
    print(f"Embedding and uploading {len(documents):,} documents (this can take a while) ...")
    text_vectordb.upload_documents(documents, ids, batch_size)

    # Save the category names for the agent (it uses them to check the image category).
    save_categories(categories)

    print(f"Done: {len(documents):,} documents in collection '{text_vectordb.COLLECTION_NAME}'.")
    print(f"Saved {len(categories)} category names to {CATEGORIES_FILE}.")
    return len(documents)


def show_examples(documents, how_many=3):
    """Print a few documents so you can SEE what will be uploaded."""
    for document in documents[:how_many]:
        print("-" * 70)
        print("page_content (the text that gets embedded):")
        print("   ", document.page_content)
        print("metadata (the facts we can filter on):")
        for key, value in document.metadata.items():
            print(f"    {key}: {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Put Olist review comments into Qdrant")
    parser.add_argument("--data-dir", default=".", help="folder with the Olist CSV files")
    parser.add_argument("--dry-run", action="store_true",
                        help="only build the documents and show examples; no Qdrant, no model")
    parser.add_argument("--limit", type=int, default=None,
                        help="index only the first N documents (quick test)")
    parser.add_argument("--no-recreate", action="store_true",
                        help="keep the existing collection and add to it")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    if args.dry_run:
        documents, ids, categories = text_combinedata.build_documents(args.data_dir)
        show_examples(documents)
        print("-" * 70)
        print(f"Dry run finished. {len(documents):,} documents, {len(categories)} categories. "
              "Nothing was uploaded.")
    else:
        ingest(args.data_dir, recreate=not args.no_recreate,
               batch_size=args.batch_size, limit=args.limit)


if __name__ == "__main__":
    main()