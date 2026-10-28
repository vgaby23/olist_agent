import re


STARTER_QUESTIONS = {
    "customer": [
        "What are the 5 best-rated sellers for health & beauty?",
        "What do reviewers complain about most in bed_bath_table?",
        "Which cheap toys get the best reviews?",
        # "What's the average delivery time to RJ?",
        "What do reviewers say about the top-ranked seller for watches?",
    ],
    "seller": [
        "How has my monthly revenue changed over the last 12 months?",
        "What are the main complaints in my 1-2 star reviews?",
        "How does my average rating compare to the benchmark for my category?",
        "Which states do most of my customers come from?",
        "What's my average delivery time and late-delivery rate?",
    ],
    "company": [
        "What was the total GMV per month in 2018?",
        "Which categories have the lowest average review scores?",
        "Top 10 sellers by revenue in health_beauty",
        "Which states have the slowest deliveries and the most late orders?",
        "What payment methods do customers use, and how many installments on average?",
    ],
}

HIDDEN_COLUMNS = re.compile(r"customer_(unique_)?id|zip", re.IGNORECASE)
TIME_COLUMN = re.compile(r"month|year|date|week|day", re.IGNORECASE)
MARKDOWN_TABLE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$",
    re.MULTILINE,
)
MAX_TABLES = 3
