from langchain_core.messages import AIMessage

from utils.chat_config import HIDDEN_COLUMNS, MAX_TABLES


def latest_versions(tables):
    """Keep the latest table when the same columns appear more than once."""
    return [
        table
        for number, table in enumerate(tables)
        if not any(
            list(later.columns) == list(table.columns)
            for later in tables[number + 1:]
        )
    ]


def answer_tables(results, role):
    """Return displayable SQL results, hiding customer identifiers outside company access."""
    tables = []
    for result in results:
        data = result.get("data")
        if data is None or data.empty:
            continue
        if role != "company":
            data = data[
                [column for column in data.columns if not HIDDEN_COLUMNS.search(str(column))]
            ]
        if len(data.columns):
            tables.append(data.reset_index(drop=True))
    return latest_versions(tables)[-MAX_TABLES:]


def collect_token_usage(messages):
    """Sum token usage reported by model messages generated for one answer."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    has_usage = False
    for message in messages:
        if not isinstance(message, AIMessage) or not message.usage_metadata:
            continue
        usage = message.usage_metadata
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["total_tokens"] += usage.get(
            "total_tokens", input_tokens + output_tokens
        )
        has_usage = True
    return totals if has_usage else None
