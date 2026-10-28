import re

import streamlit as st
import altair as alt
from utils.chat_config import STARTER_QUESTIONS, TIME_COLUMN
from utils.chat_history import ask_chip
from utils.chat_text import clean_answer
from utils.chat_results import latest_versions


UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")


def markdown_safe(text):
    """Escape $ so Streamlit does not render the text between two prices ("R$13,237 ... R$13,169") as LaTeX math."""
    return UNESCAPED_DOLLAR.sub(r"\\$", text or "")


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
    st.info(markdown_safe(clean_answer(message.text)), icon="🔒")
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

