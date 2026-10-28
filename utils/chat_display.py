import re

import streamlit as st
import altair as alt
from agent import supervisor_agent
from langchain_core.messages import AIMessage
from utils.chat_config import HIDDEN_COLUMNS, MAX_TABLES, STARTER_QUESTIONS, TIME_COLUMN
from utils.chat_history import ask_chip


UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")


def markdown_safe(text):
    """Escape $ so Streamlit does not render the text between two prices ("R$13,237 ... R$13,169") as LaTeX math."""
    return UNESCAPED_DOLLAR.sub(r"\\$", text or "")


def latest_versions(tables):
    """The data agent can be asked twice or retry a query: of tables with the same columns, keep only the last."""
    return [table for number, table in enumerate(tables)
            if not any(list(later.columns) == list(table.columns) for later in tables[number + 1:])]


def answer_tables(results, role):
    """Non-empty SQL results behind an answer, without customer-identifying columns for non-company roles."""
    tables = []
    for result in results:
        data = result.get("data")
        if data is None or data.empty:
            continue
        if role != "company":
            data = data[[column for column in data.columns if not HIDDEN_COLUMNS.search(str(column))]]
        if len(data.columns):
            tables.append(data.reset_index(drop=True))
    return latest_versions(tables)[-MAX_TABLES:]



def make_chart(data):
    """A line chart for a trend (month/year/date column), a bar chart for a ranking, None when neither fits."""
    if not 2 <= len(data) <= 60:
        return None
    time_columns = [column for column in data.columns if TIME_COLUMN.search(str(column))]
    label_columns = [column for column in data.columns if column not in data.select_dtypes("number").columns]
    x = time_columns[0] if time_columns else (label_columns[0] if len(label_columns) == 1 else None)
    metrics = [column for column in data.select_dtypes("number").columns if column != x]
    if x is None or not metrics:
        return None
    y = metrics[0]
    if time_columns:
        return alt.Chart(data).mark_line(point=True).encode(
            x=alt.X(f"{x}:O", title=x), y=alt.Y(f"{y}:Q", title=y), tooltip=list(data.columns))
    return alt.Chart(data).mark_bar().encode(
        x=alt.X(f"{y}:Q", title=y), y=alt.Y(f"{x}:N", sort="-x", title=x), tooltip=list(data.columns))



def show_tables(tables, key, collapsed=False):
    """Each SQL result under the answer: chart and table tabs when a chart fits, plus a CSV download.
    Collapsed into an expander when the answer already shows the data as a markdown table."""
    tables = latest_versions(tables)    # also answers saved before de-duplication
    if collapsed and tables:
        with st.expander("📋 Data & CSV", expanded=False):
            show_tables(tables, key)
        return
    for number, data in enumerate(tables):
        chart = make_chart(data)
        if chart is not None:
            chart_tab, table_tab = st.tabs(["📈 Chart", "📋 Table"])
            chart_tab.altair_chart(chart, width="stretch")
        else:
            table_tab = st.container()
        table_tab.dataframe(data, hide_index=True, width="stretch")
        st.download_button("Download CSV", data.to_csv(index=False), file_name=f"olist_data_{key}_{number + 1}.csv",
                           mime="text/csv", icon="⬇️", key=f"csv_{key}_{number}")


def show_token_usage(message):
    """Show reported token usage for an answer, including in archived chat history."""
    usage = message.additional_kwargs.get("token_usage")
    if not usage:
        return
    with st.expander(f"Token usage · {usage['total_tokens']:,} total", expanded=False):
        input_metric, output_metric, total_metric = st.columns(3)
        input_metric.metric("Input", f"{usage['input_tokens']:,}")
        output_metric.metric("Output", f"{usage['output_tokens']:,}")
        total_metric.metric("Total", f"{usage['total_tokens']:,}")

def show_refusal(message, role, key, latest):
    """A refused request (agent/policy.py) as a notice, with allowed questions to try on the latest turn."""
    st.info(markdown_safe(supervisor_agent.clean_answer(message.text)), icon="🔒")
    if latest:
        st.pills("You could ask instead", STARTER_QUESTIONS.get(role, [])[:3], key=f"refusal_chip_{key}",
                 on_change=ask_chip, args=(f"refusal_chip_{key}",))



def show_answer_details(message):
    """Small caption with how long the answer took and which specialists worked on it."""
    elapsed = message.additional_kwargs.get("elapsed")
    activity = message.additional_kwargs.get("activity", [])
    specialists = [name for name in ("SQL specialist", "Review specialist")
                   if any(step.startswith(name) for step in activity)]
    details = [f"⏱ {elapsed:.0f} s"] if elapsed else []
    if specialists:
        details.append("used " + " and ".join(specialists))
    if details:
        st.caption(" · ".join(details))


def show_message_flow(entries):
    """Display the delegated specialist messages stored with an answer."""
    if not entries:
        return
    with st.expander("Agent message flow", expanded=False):
        for entry in entries:
            st.markdown(f"**{entry['sender']} -> {entry['recipient']}**")
            st.code(entry["message"], language=None)


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

