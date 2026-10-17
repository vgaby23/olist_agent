"""
text_embedding.py - STEP 3: the embedding model
===============================================
 
An embedding model turns a text into a vector (a list of numbers) that captures its MEANING.
Texts with a similar meaning get similar vectors, and that is how the search works.
 
IMPORTANT: this file is shared by ingest.py (embeds the review comments) AND by the API
(embeds the user's question). Both MUST use the same model, otherwise the vectors cannot be
compared. That is why the model is defined in exactly one place.
 
To change the model: set the environment variable EMBEDDING_MODEL, then run ingest.py again.
"""

import os
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

# The default MUST be an OpenAI model name now (the old default was a HuggingFace name).
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Mengembalikan Embedding Model untuk digunakan dalam penyimpanan vector dan RAG"""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)