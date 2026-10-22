"""
text_vectordb.py - STEP 4: Qdrant (the vector database)
=======================================================

What is here:
    get_client()          connect to Qdrant                          (used by ingest AND the API)
    get_store()           LangChain's wrapper to SEARCH the collection   (used by the API)
    ensure_collection()   create the collection + indexes            (used by ingest)
    upload_documents()    embed the documents and upload them        (used by ingest)

Where Qdrant lives is set with environment variables:
    QDRANT_URL       e.g. https://xxxx.cloud.qdrant.io:6333   (default http://localhost:6333)
    QDRANT_API_KEY   your key (leave empty for a local Qdrant)
    QDRANT_PATH      a local folder instead of a server (development only)
    COLLECTION_NAME  default "olist_reviews"
"""
import os
from functools import lru_cache

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from dataset.preprocessing.vector_data import text_embedding

COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "olist_reviews")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
QDRANT_PATH = os.environ.get("QDRANT_PATH") or None
    
# LangChain saves our metadata under the name "metadata" inside Qdrant, so a field called
# "review_score" is called "metadata.review_score" in Qdrant filters and indexes.
META = "metadata."


@lru_cache(maxsize=1)      # connect once, then reuse the same connection
def get_client():
    if QDRANT_PATH:                      # a local folder: only one program can use it at a time
        return QdrantClient(path=QDRANT_PATH)
    if QDRANT_URL == ":memory:":         # lives in RAM, gone when the program stops (used by the tests)
        return QdrantClient(":memory:")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)


@lru_cache(maxsize=1)
def get_store():
    """Connect to the EXISTING collection (ingest.py creates it). Used for searching."""
    return QdrantVectorStore(
        client=get_client(),
        collection_name=COLLECTION_NAME,
        embedding=text_embedding.get_embeddings(),
    )


def ensure_collection(recreate=True):
    """Make sure the collection exists (a collection is like a table).
    recreate=True deletes the old one first, so you start clean."""
    client = get_client()
    name = COLLECTION_NAME

    if recreate and client.collection_exists(name):
        client.delete_collection(name)

    if not client.collection_exists(name):
        # The vector size depends on the embedding model, so we ask the model:
        # embed one test sentence and count the numbers.
        vector_size = len(text_embedding.get_embeddings().embed_query("test"))
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,   # compare vectors by the angle between them
            ),
        )

    # A "payload index" makes filtering on a field fast (like an index in SQL).
    fields_to_index = [
        ("review_score", models.PayloadSchemaType.INTEGER),
        ("price", models.PayloadSchemaType.INTEGER),
        ("seller_city", models.PayloadSchemaType.KEYWORD),
        ("seller_id", models.PayloadSchemaType.KEYWORD),
        ("product_name", models.PayloadSchemaType.KEYWORD),
        ("product_id", models.PayloadSchemaType.KEYWORD),
        ("order_id", models.PayloadSchemaType.KEYWORD),
    ]
    for field_name, field_type in fields_to_index:
        client.create_payload_index(
            collection_name=name,
            field_name=META + field_name,
            field_schema=field_type,
        )


def upload_documents(documents, ids, batch_size=64):
    """Embed the documents and upload them to Qdrant, batch_size at a time.

    add_documents() does, for every batch:
       1. turn the comments into vectors (using the embedding model)
       2. upload vector + comment + metadata to Qdrant"""
    store = QdrantVectorStore(
        client=get_client(),
        collection_name=COLLECTION_NAME,
        embedding=text_embedding.get_embeddings(),
    )
    store.add_documents(documents, ids=ids, batch_size=batch_size)