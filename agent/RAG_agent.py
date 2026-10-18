"""
rag_agent.py - the REVIEW (RAG) agent
=====================================

WHAT IT DOES
    Answers questions about what customers SAY in their reviews: complaints, praise, reasons for a
    bad rating. It searches the review comments in Qdrant (through rag_core.py) and writes a short
    answer with example quotes. It does NOT do numbers (counts, averages): that is the SQL agent's job.

HOW THE SUPERVISOR USES IT   (note for the teammate who builds the supervisor)
    The simplest way, one line - the agent is already wrapped as a tool:

        from rag_agent import ask_review_agent
        supervisor = create_agent(llm, [ask_review_agent, <the SQL agent tool>], system_prompt=...)

    Or call it as a normal function (question in, text out, no memory between calls):

        from rag_agent import ask_rag_agent
        text = ask_rag_agent("What do customers complain about in health_beauty?")

    Rules for the supervisor:
      * Send a COMPLETE question. This agent does not see the conversation, so write the category,
        rating or city into the question itself.
      * Say in the question which LANGUAGE the user wrote in (e.g. "Answer in Bahasa Indonesia"). The
        reviews are Portuguese; every quote comes back as the Portuguese original PLUS a translation
        into that language. Without a hint, the agent uses the language of the question you send it.
      * To get the review of ONE order (e.g. after the SQL agent found its order_id), put the order_id in the
        question: "What did order <order_id> say?". Price limits ("above 150 BRL") also work as filters.
      * For a SELLER user, put their seller_id in the question: "Why do customers of seller <seller_id> give
        low ratings?". The same works for one product: "What do reviews of product <product_id> say?".
      * The answer is plain text: a summary plus example quotes, each with its rating, product category, seller city
        and order_id.
        order_id links to the SQL tables, so the SQL agent can look up more about that order.
      * If something goes wrong the answer is a sentence starting with "The review search failed" or
        similar, never an exception.

WHAT IT NEEDS (in the .env file)
    OPENAI_API_KEY            the LLM and the embedding model
    QDRANT_URL, QDRANT_API_KEY
    LLM_MODEL                 optional, default gpt-4o-mini
    CATEGORIES_FILE           optional, path to categories.json (written by the ingest step)

TRY IT
    python rag_agent.py                                  chat in the terminal, "exit" to stop
    python rag_agent.py "why do people give 1 star?"     one question
    python rag_agent.py --show-steps                     also print every search the agent runs

THE FILE HAS 4 PARTS
    1. Settings and finding your files
    2. The search tool
    3. The agent (instructions + build_rag_agent)
    4. Ways to use it: ask_rag_agent, ask_review_agent (for the supervisor), chat, main
"""
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# MUST run before "import rag_core": the files it imports read the .env values when they are imported.
load_dotenv()

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# =============================================================================
# Part 1: SETTINGS AND FINDING YOUR FILES
# =============================================================================
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")      # the OpenAI chat model used as the "brain"

# The "match" score of a review is a cosine similarity: higher = closer to the question.
# Tuned on the full index (34,732 reviews) with test_folder/tune_thresholds.py:
#   real topics, Portuguese query      0.63 to 0.92      real topics, English query     0.46 to 0.68
#   nonsense / near-miss, Portuguese   up to 0.56        nonsense, English              up to 0.41
# No single number separates them, so: below MIN_MATCH a review is dropped, below WEAK_MATCH it is
# marked "(weak)" and the LLM must read it carefully. The agent is told to search in Portuguese.
SEARCH_POOL = 50                                            # how many reviews are checked when must_contain is used
MIN_MATCH = float(os.environ.get("MIN_MATCH", "0.45"))      # below this a review is dropped (0 = keep everything)
WEAK_MATCH = float(os.environ.get("WEAK_MATCH", "0.60"))    # below this a review is marked "(weak)"

# General queries used when the FILTERS are the topic ("low-rated health_beauty products").
# They score low (0.25 to 0.55) even when every review fits, so the score cut-off is skipped for them.
GENERAL_QUERIES = {"reclamação", "reclamacao", "elogio", "opinião sobre o produto", "opiniao sobre o produto",
                   "avaliação", "avaliacao", "complaint", "complaints", "praise", "review", "reviews"}


def is_general_query(query):
    """True for the general words above, e.g. "Reclamação " -> True, "atraso na entrega" -> False."""
    return query.strip().lower() in GENERAL_QUERIES

THIS_FOLDER = Path(__file__).resolve().parent

from agent.language import guess_language
from data_preprocessing.vector_data import rag_core

def add_language_hint(question):
    """Add one sentence telling the agent which language the user wrote in.
    Not added when the question already says "Answer in ..." (a supervisor does that) or when unclear."""
    if "answer in" in question.lower():
        return question
    language = guess_language(question)
    if language is None:
        return question
    return (f'{question}\n\n(The user wrote in {language}: use answer_language="{language}" '
            f"and write the whole answer in {language}.)")


# Questions about how often or how many cannot be answered from a handful of search results.
COUNT_PHRASES = ("most common", "most frequent", "most often", "how many", "how often", "percentage", "what percent",
                 "average", "paling sering", "paling umum", "berapa banyak", "berapa jumlah", "persentase",
                 "rata-rata", "mais comum", "mais frequente", "quantos", "quantas", "porcentagem", "média")


def add_counts_hint(question):
    """For "what is the most common ..." questions, remind the agent that it cannot count."""
    lowered = question.lower()
    if any(phrase in lowered for phrase in COUNT_PHRASES):
        return (f"{question}\n\n(This asks how often or how many. A search cannot answer that: do NOT claim "
                f"what is most common or give any number. Say so, then show at most three examples, clearly "
                f"labelled as examples, and say the SQL specialist must do the counting.)")
    return question


def prepare_question(question):
    """Everything we add to the user's question before the agent sees it."""
    return add_counts_hint(add_language_hint(question))


def explain_error(error):
    """Turn an exception into one readable sentence, with the real cause and a hint when it is a network problem."""
    text = f"{type(error).__name__}: {str(error)[:200]}"
    cause = getattr(error, "__cause__", None)
    if cause is not None:                       # e.g. the low-level reason behind "Connection error"
        text += f" (cause: {type(cause).__name__}: {str(cause)[:150]})"
    name = type(error).__name__
    if "Connection" in name or "Timeout" in name:
        text += (". This looks like a network problem between this computer and OpenAI (internet, VPN, proxy "
                 "or an OpenAI outage), not a problem with the question. Check the connection and try again.")
    return text


def retrieve(query, top_k, filters):
    """The ONE place where this agent talks to the search code (rag_core.py, which searches Qdrant).
    Returns a list of {"content", "relevance_score", "metadata"}."""
    return rag_core.search(query, top_k=top_k, filters=filters)


def find_categories():
    """Load the category names saved by the ingest step (categories.json). Empty list if not found."""
    path = os.environ.get("CATEGORIES_FILE")
    if not path:
        for folder in (Path.cwd(), THIS_FOLDER, THIS_FOLDER.parent, THIS_FOLDER.parent.parent):
            if (folder / "categories.json").exists():
                path = str(folder / "categories.json")
                break
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    return []


CATEGORIES = find_categories()


def format_categories(names):
    """["housewares", "housewares", "toys"] -> "housewares x2, toys".
    An order can contain many items, so the same category may appear several times. We show each
    category once, with a count when it is more than one item (easier to read and to verify)."""
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1          # dictionaries keep the order of first appearance
    parts = []
    for name, count in counts.items():
        parts.append(name if count == 1 else f"{name} x{count}")
    return ", ".join(parts)


# =============================================================================
# Part 2: THE SEARCH TOOL
# =============================================================================
# The @tool line turns a normal Python function into something the LLM can call.
# IMPORTANT: the LLM never sees our code. It only sees the function NAME, the argument names and
# the text inside the triple quotes (the "docstring"). So the docstring is an instruction manual
# written for the LLM. Vague docstring = tool used badly.
@tool
def search_reviews(
    query: str,
    answer_language: str,
    how_many: int = 5,
    max_rating: int | None = None,
    min_rating: int | None = None,
    product_category: str | None = None,
    seller_id: str | None = None,
    product_id: str | None = None,
    seller_city: str | None = None,
    order_id: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    must_contain: str | None = None,
) -> str:
    """Search customer review comments by MEANING and return the best matches.

    Use this to find out what customers say: complaints, praise, reasons for a bad rating,
    opinions about a product type.

    The reviews are written in Portuguese, so write the query in Portuguese (it matches much better).

    Arguments:
        query: what to look for, in Portuguese, e.g. "atraso na entrega" or "produto quebrado".
        answer_language: the language the USER wrote in (or asked for), e.g. "English", "Bahasa Indonesia"
            or "Portuguese". Decide this first. The tool reminds you of it when it returns the reviews.
        how_many: number of reviews to return, 1 to 10 (default 5).
        max_rating: only reviews with this rating or lower (1 = worst, 5 = best). Use 2 for complaints.
        min_rating: only reviews with this rating or higher. Use 4 for praise.
        product_category: only reviews of orders that contain this ENGLISH category, e.g. "health_beauty".
        seller_city: only reviews of orders from sellers in this city, e.g. "sao paulo".
        order_id: the review of ONE specific order (the long id). Use it with a general query like "avaliação".
        seller_id: only reviews of orders from ONE seller (the long seller id, copied exactly). Use it when the question is about a specific seller or shop. With no other topic, use a general query like "reclamação".
        product_id: only reviews of orders that contain ONE specific product (the long product id, copied exactly).
        min_price: only orders with an item that costs at least this many Brazilian reais (BRL).
        max_price: only orders with an item that costs at most this many reais (BRL).
        must_contain: a regular expression the review text must match (case-insensitive). Use it when the user
            wants reviews that contain something specific, e.g. "[0-9]{1,2}/[0-9]{1,2}" for a date or
            "sedex|correios" for words. Only the 50 closest reviews are checked.
    """
    # 1) Turn the simple arguments into the filter dictionary that rag_core.search understands.
    filters = {}

    rating_range = {}
    if min_rating is not None:
        rating_range["gte"] = min_rating          # gte = greater than or equal
    if max_rating is not None:
        rating_range["lte"] = max_rating          # lte = less than or equal
    if rating_range:
        filters["review_score"] = rating_range

    if product_category:
        filters["product_name"] = product_category    # in the metadata, "product_name" holds the category
    if seller_city:
        filters["seller_city"] = seller_city
    if order_id:
        filters["order_id"] = order_id.strip()
    if seller_id:
        filters["seller_id"] = seller_id.strip()
    if product_id:
        filters["product_id"] = product_id.strip()

    price_range = {}
    if min_price is not None:
        price_range["gte"] = min_price
    if max_price is not None:
        price_range["lte"] = max_price
    if price_range:
        filters["price"] = price_range

    how_many = max(1, min(int(how_many), 10))         # keep it between 1 and 10

    pattern = None
    fetch = how_many                                  # how many reviews we ask the search for
    if must_contain:
        try:
            pattern = re.compile(must_contain[:100], re.IGNORECASE)
        except re.error as error:
            return f"The must_contain pattern is not a valid regular expression: {error}"
        fetch = max(how_many, SEARCH_POOL)            # look at more reviews, then keep those that match

    # 2) Do the search. If something goes wrong we RETURN the problem as text instead of crashing,
    #    so the LLM can read it and try again (or tell the user).
    try:
        results = retrieve(query, fetch, filters)
    except rag_core.FilterError as error:
        return f"The filters were not valid: {error}"
    except Exception as error:
        return f"The review search failed: {type(error).__name__}: {str(error)[:200]}"

    if not results:
        return "No reviews found for this search. Try fewer filters or different words."

    # Drop reviews that are clearly not about the topic. The search ALWAYS returns the closest reviews,
    # even for a nonsense question, so without this the LLM would see unrelated reviews and might list them.
    best_score = max(hit["relevance_score"] for hit in results)
     # The score means nothing when we look up one order, or when the filters are the topic (general query).
    use_score = not order_id and not is_general_query(query)
    if use_score:
        results = [hit for hit in results if hit["relevance_score"] >= MIN_MATCH]
    if not results:
        return (f"No relevant reviews found: the closest ones scored only {best_score:.2f} "
                f"(below {MIN_MATCH:.2f}), so they are not about this topic. "
                f"Tell the user that nothing relevant was found, and do not list any reviews.")

    # Keep only the reviews whose text contains what the user asked for (checked here, not by the LLM).
    checked = 0
    if pattern is not None:
        checked = len(results)
        results = [hit for hit in results if pattern.search(hit["content"])][:how_many]
        if not results:
            return (f"None of the {checked} closest reviews contains '{must_contain}'. "
                    f"Tell the user that no such review was found among the closest ones. Do not list other reviews.")

    # 3) Turn the results into plain text for the LLM to read.
    lines = []
    for number, hit in enumerate(results, start=1):
        meta = hit["metadata"]
        categories = format_categories(meta.get("product_name", []))
        prices = ", ".join(str(price) for price in meta.get("price", []))
        weak = " (weak)" if hit["relevance_score"] < WEAK_MATCH and use_score else ""
        lines.append(
            f"{number}) match {hit['relevance_score']:.2f}{weak} | rating {meta.get('review_score')}/5 | "
            f"category: {categories} | price: {prices} BRL | city: {meta.get('seller_city')} | "
            f"order_id: {meta.get('order_id')}\n"
            f"   \"{hit['content']}\""
        )

    if pattern is not None:
        lines.append(f"\nMATCHING REVIEWS FOUND: {len(results)} (out of the {checked} closest reviews). Tell the "
                     f"user exactly this number. If they asked for more, say that fewer were found.")

    has_price_filter = min_price is not None or max_price is not None
    if has_price_filter:
        limits = []
        if min_price is not None:
            limits.append(f"at least {min_price:g}")
        if max_price is not None:
            limits.append(f"at most {max_price:g}")
        lines.append(f"\nPRICE FILTER USED: an item price of {' and '.join(limits)} BRL. Tell the user which price "
                     f"limit you used, and add each quote's price to the parentheses (price: ... BRL).")

    # When EVERY review is weak, the search probably found nothing about the topic. Say it here in code:
    # with only the prompt, the LLM still listed weak reviews (tested: "bitcoin", "chocolate cake").
    if use_score and all(hit["relevance_score"] < WEAK_MATCH for hit in results):
        lines.append("\nALL MATCHES ARE WEAK: none of these reviews is clearly about the question. Read each one and "
                     "quote ONLY reviews that really answer it. If none does, say that nothing relevant was found "
                     "and do NOT list any review, not even as an example.")

    # The LAST thing the LLM reads before it writes its answer is a reminder of the answer language.
    # (Without this, the LLM tends to slip back into English after reading the results.)
    fields = ["rating", "category", "price", "city", "order_id"] if has_price_filter else \
             ["rating", "category", "city", "order_id"]
    if "portugu" in answer_language.lower():
        lines.append("\nANSWER LANGUAGE: Portuguese. Show the original quotes only, no translation. "
                     f"Still show {', '.join(fields[:-1])} and {fields[-1]}.")
    else:
        lines.append(
            f"\nANSWER LANGUAGE: {answer_language}. Write your summary, your closing sentence and the translation "
            f"of EVERY quote in {answer_language}, even if these results look English. Show each quote as: "
            f"\"original Portuguese\" -> \"translation in {answer_language}\" ({', '.join(fields)}). "
            f"Copy the category, the price and the city exactly as shown in the results."
        )
    return "\n".join(lines)


# =============================================================================
# Part 3: THE AGENT
# =============================================================================
def make_system_prompt():
    """The system prompt = the instructions the LLM reads before every question."""
    prompt = """You are the REVIEW specialist for Olist, a Brazilian online marketplace.
You have one tool: search_reviews. Use it to find out what customers say.

LANGUAGE (the most important rule)
- The reviews are in Portuguese. Your ANSWER LANGUAGE is the language the question is written in
  (English, Bahasa Indonesia, ...), or the language the question asks for. Decide it first and pass it
  to search_reviews as answer_language.
- Write EVERYTHING you say in the answer language: the summary, the translations and the closing sentence.
  Never switch to English because the tool results look English.
- Show every quote in BOTH languages: the Portuguese original exactly as written, then the translation,
  then the rating, the product category, the seller city and the order_id. Translate faithfully.
  If the answer language is Portuguese, show the original only (still with rating, category, city and order_id).
- Copy the product category ("category: ...") and the seller city ("city: ...") exactly as given in the
  search results. Do not translate or guess them. If an order has several categories, show all of them.

Example, question in Bahasa Indonesia (answer_language "Bahasa Indonesia"):
  Berikut beberapa review tentang produk rusak:
  1. "produto deu defeito na 1 vez de uso." -> "produk rusak pada pemakaian pertama." (rating 1/5, category: health_beauty, city: sao paulo, order_id abc123)
  Review-review ini menunjukkan ketidakpuasan terhadap kualitas produk.

Example, question in English (answer_language "English"):
  Here are some reviews about damaged products:
  1. "produto deu defeito na 1 vez de uso." -> "the product broke on the first use." (rating 1/5, category: health_beauty, city: sao paulo, order_id abc123)
  These reviews show dissatisfaction with product quality.

HOW TO WORK
- Write the search query in PORTUGUESE, even when the user writes in another language: the reviews are
  Portuguese and Portuguese queries match much better (e.g. "atraso na entrega", not "late delivery").
- Use the filters when the question mentions a rating, a product category or a seller city.
  Complaints usually mean max_rating=2, praise means min_rating=4.
- The query is only the TOPIC, e.g. "produto quebrado" or "atraso na entrega". Do NOT repeat a category,
  city or rating in the query: use the filters for those.
  KEEP the topic of the question in the query: "praise about fast shipping" -> query "entrega rápida" with
  min_rating=4, NOT "elogio". "complaints about late delivery" -> query "atraso na entrega" with max_rating=2.
  Only If the question has NO topic besides the filters (e.g. "low-rated health_beauty products"), use a general
  query: "reclamação" for low ratings, "elogio" for high ratings, "opinião sobre o produto" otherwise.
  Never use a category name such as "health_beauty" as the query.
- "worst" or "lowest rated" means max_rating=1.
- Only add a filter when the question asks for it. Use a rating filter only for complaints, praise or stars.
- Price: "expensive", "cheap" or an amount -> min_price / max_price (Brazilian reais). Pick a sensible limit
  (e.g. min_price=150 for "expensive", max_price=50 for "cheap") and tell the user which limit you used.
- A specific order -> order_id, with the general query "avaliação". The search cannot find an order any other way.
- A specific seller -> seller_id; a specific product -> product_id (copy the long id exactly). If the question has no other topic, use a general query: "reclamação" for low ratings, "elogio" for high ratings.
- If the first search is not useful, try again with different words or fewer filters.
- Reply with: 1) a short summary of the main themes, 2) two to four example quotes.
- RELEVANCE: the search always returns the closest reviews, even when none is really about the topic.
  Read each review and quote only those that truly answer the question. Each result has a match score
  (higher = closer) that supports your judgement but does not replace it. If only some are relevant, quote
  those and say the others were about something else. If NONE is relevant, say plainly that nothing relevant
  was found and do NOT list the reviews. "(weak)" is only a hint: read the text. A weak match can still be
  relevant and a strong one can be off-topic. (When the question is only about filters, the score matters less.)
- CONTENT CHECK: when the user wants reviews that contain something specific (a date, a name, a word), use
  must_contain (a regular expression, e.g. "[0-9]{1,2}/[0-9]{1,2}" for a date). If fewer than requested come
  back, say how many you found. Never pad the list with reviews that do not fit.
- COUNTS: a search only returns the few closest reviews, so you cannot know what is "the most common" complaint
  or "how many" reviews say something. For such questions say so, show a few examples clearly labelled as
  examples, and say that counts must come from the SQL specialist.
- QUOTES: copy the Portuguese exactly. In the translation keep dates as written (14/02 stays 14/02), keep
  store names as written (even odd ones like "stark" or "lannister") and do not drop any words.
- Use EXACTLY this layout for every quote (no colon after order_id):
  "original" -> "translation" (rating 1/5, category: housewares x2, city: sao paulo, order_id abc123)
- Never invent quotes. If nothing is found, say so, and mention what you tried.
- You cannot count or average anything. If the question needs numbers, say that the SQL specialist should answer that part.
"""
    if CATEGORIES:
        prompt += "\nPRODUCT CATEGORIES (exact English names for product_category):\n"
        prompt += ", ".join(CATEGORIES) + "\n"
    return prompt


def build_rag_agent(llm=None):
    """Put the LLM, the tool and the instructions together. Returns the agent.

    `llm` is optional: normally we create the OpenAI model here. (The tests pass in a fake one.)
    """
    if llm is None:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)      # temperature 0 = least random answers
    return create_agent(llm, [search_reviews], system_prompt=make_system_prompt(), name="review_agent")


# =============================================================================
# Part 4: WAYS TO USE THE AGENT
# =============================================================================
_agent = None          # built on first use and then reused (building it each time would be slow)


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_rag_agent()
    return _agent


def ask_rag_agent(question, show_steps=False):
    """One question in, one text answer out. No memory between calls.

    This is the function a supervisor can call.
    """
    try:
        result = get_agent().invoke({"messages": [{"role": "user", "content": prepare_question(question)}]})
    except Exception as error:         # return the problem as text so the caller never crashes
        return f"The review agent failed: {explain_error(error)}"

    if show_steps:
        for message in result["messages"]:
            for call in getattr(message, "tool_calls", None) or []:
                print(f"   [tool call] {call['name']}({call['args']})")
            if message.type == "tool":                       # what the tool sent back to the LLM
                shown = str(message.content)
                print("   [tool result]\n" + "\n".join("      " + line for line in shown[:3000].splitlines()))

    return result["messages"][-1].content


# The same function, packaged as a TOOL, so a supervisor agent can use it directly.
# (Again the docstring is what the supervisor's LLM reads.)
@tool
def ask_review_agent(question: str) -> str:
    """Ask the review specialist what customers say: complaints, praise, reasons for a rating.

    It searches the Portuguese review comments and returns a summary with example quotes, each shown
    in the original Portuguese AND in a translation (with rating, product category, seller city and order_id).
    It cannot count or average anything.

    Arguments:
        question: a complete, self-contained question. ALWAYS say which language the answer must be in
            (the language the user wrote in), e.g.
            "What do customers complain about in the health_beauty category (rating 1-2)? Answer in Bahasa Indonesia."
    """
    return ask_rag_agent(question)


def chat(show_steps=False):
    """A simple loop in the terminal: you type, the agent answers. Each question is answered on its own."""
    print("Olist review agent. Type your question, or 'exit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue
        print("Bot:", ask_rag_agent(question, show_steps), "\n")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)

    show_steps = "--show-steps" in args
    if show_steps:
        args.remove("--show-steps")

    if args:                                          # a question was given: answer once and stop
        print(ask_rag_agent(" ".join(args), show_steps))
    else:
        chat(show_steps)


if __name__ == "__main__":
    main()