"""
agent_client.py - the small HTTP client the agent uses to talk to the RAG API
=============================================================================

The RAG API (app.py, running in Docker / Cloud Run) does the Qdrant search.
This file only SENDS the question there and READS the answer back. It needs no Qdrant code.

THE API CONTRACT (do not change)
    POST /retrieve   body:   {"query": "...", "top_k": 5, "filters": {...}}
                     answer: {"status": "success" | "failed",
                              "retrieved_documents": [{"content", "relevance_score", "metadata"}],
                              "error_message": None or "..."}
    GET  /health     answer: 200 if the API is up

    HTTP codes: 200 ok, 400 bad filter, 422 bad body, 500 server error

TRY IT (the API must be running, e.g. docker run ... -p 8080:8080 olist-rag)
    from agent import agent_client
    print(agent_client.health("http://localhost:8080"))
    docs = agent_client.retrieve("entrega atrasada", top_k=3, base_url="http://localhost:8080")
    print(docs[0]["content"])

CLOUD RUN WITH LOGIN REQUIRED (--no-allow-unauthenticated)
    Set RAG_AUTH_AUDIENCE to the Cloud Run URL and install google-auth.
    This file then adds a Google ID token to every request. Without RAG_AUTH_AUDIENCE no token is sent.
"""
import os

import httpx

RAG_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8080")
RAG_AUTH_AUDIENCE = os.environ.get("RAG_AUTH_AUDIENCE", "").strip()
TIMEOUT_SECONDS = float(os.environ.get("RAG_TIMEOUT", "30"))


# ---------------------------------------------------------------------------------------------
# PART 1. The two errors the agent can catch
# ---------------------------------------------------------------------------------------------
class RagError(Exception):
    """The API could not be reached, or it answered with an error."""


class FilterRejected(RagError):
    """The API said the filters were wrong (HTTP 400). The agent can fix the filters and try again."""


# ---------------------------------------------------------------------------------------------
# PART 2. Small helpers
# ---------------------------------------------------------------------------------------------
def make_url(base_url, path):
    """Join "http://host:8080/" and "/retrieve" into "http://host:8080/retrieve"."""
    base_url = (base_url or RAG_API_URL).rstrip("/")
    return base_url + path


def make_headers():
    """Headers for every request. Adds a Google ID token only when RAG_AUTH_AUDIENCE is set."""
    headers = {"Content-Type": "application/json"}
    if RAG_AUTH_AUDIENCE:
        # imported here so google-auth is only needed when Cloud Run requires a login
        import google.auth.transport.requests
        import google.oauth2.id_token

        request = google.auth.transport.requests.Request()
        token = google.oauth2.id_token.fetch_id_token(request, RAG_AUTH_AUDIENCE)
        headers["Authorization"] = f"Bearer {token}"
    return headers


def read_error_message(response):
    """Get the API's error text out of a response, whatever shape it has."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200] or f"HTTP {response.status_code}"
    if isinstance(body, dict):
        # our own answer uses "error_message"; FastAPI's 422 answer uses "detail"
        return str(body.get("error_message") or body.get("detail") or body)[:200]
    return str(body)[:200]


# ---------------------------------------------------------------------------------------------
# PART 3. The two calls
# ---------------------------------------------------------------------------------------------
def retrieve(query, top_k=5, filters=None, base_url=None):
    """Ask the API for the reviews that best match the query.

    Returns the same list as rag_core.search(): [{"content", "relevance_score", "metadata"}, ...]
    Raises FilterRejected for a bad filter (HTTP 400), RagError for anything else that went wrong.
    """
    body = {"query": query, "top_k": top_k, "filters": filters or {}}

    # 1. Send the request. No connection at all (wrong URL, API down, no internet) -> RagError.
    try:
        response = httpx.post(make_url(base_url, "/retrieve"), json=body,
                              headers=make_headers(), timeout=TIMEOUT_SECONDS)
    except httpx.HTTPError as error:
        raise RagError(f"Could not reach the RAG API: {type(error).__name__}: {error}") from error

    # 2. Check the HTTP code
    if response.status_code == 400:
        raise FilterRejected(read_error_message(response))
    if response.status_code != 200:
        raise RagError(f"RAG API answered HTTP {response.status_code}: {read_error_message(response)}")

    # 3. Read the answer. The API can also say "failed" inside a 200 answer.
    answer = response.json()
    if answer.get("status") != "success":
        raise RagError(f"RAG API failed: {answer.get('error_message') or 'no reason given'}")
    return answer.get("retrieved_documents", [])


def health(base_url=None):
    """True if GET /health answers 200, False otherwise (never raises)."""
    try:
        response = httpx.get(make_url(base_url, "/health"), headers=make_headers(), timeout=5)
        return response.status_code == 200
    except Exception:
        return False
