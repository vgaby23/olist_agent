from dotenv import load_dotenv
load_dotenv()

from dataset.preprocessing.vector_data import rag_core
from dataset.preprocessing.vector_data import text_vectordb

client = text_vectordb.get_client()
name = text_vectordb.COLLECTION_NAME

print("points in collection :", client.count(name).count)
print("vector size          :", client.get_collection(name).config.params.vectors.size)

points, _ = client.scroll(name, limit=3, with_payload=True, with_vectors=False)
for p in points:
    print("-" * 60)
    print(p.payload["page_content"])
    print(p.payload["metadata"])

print("=" * 60)
print("Search 'Broken Product' (rating 1-2 only):")
for hit in rag_core.search("produk rusak", top_k=3, filters={"review_score": {"lte": 2}}):
    print(round(hit["relevance_score"], 3), hit["metadata"]["review_score"], "|", hit["content"][:60])