import logging
import re
import time
import altair as alt
import streamlit as st
import tempfile
from pathlib import Path
from types import SimpleNamespace
from agent import supervisor_agent, sql_agent
from utils.auth import login_page
from utils import input_preprocessing
from langchain_core.messages import HumanMessage, AIMessage

# Print our own INFO logs (e.g. the voice transcript) in the terminal running streamlit
logging.basicConfig(format="%(asctime)s %(name)s: %(message)s")
logging.getLogger("utils").setLevel(logging.INFO)

# Company and seller answers often come with tables, which need the room; the shopper chat stays narrow
WIDE_ROLES = {"company", "seller"}
st.set_page_config(page_title="Olist Virtual Assistant", page_icon="🛍️",
                   layout="wide" if st.session_state.get("role") in WIDE_ROLES else "centered")

st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("role", None)
st.session_state.setdefault("username", None)
st.session_state.setdefault("messages", [])

AVATARS = {"user": "🧑", "assistant": "🛍️"}

ROLE_INTRO = {
    "customer": "Hi! I can help you find well-rated products and sellers, tell you what other shoppers say in "
                "their reviews, and how long deliveries usually take to your state.",
    "seller": "Hi! I can analyse your shop's sales, revenue, deliveries and reviews, and compare your shop with "
              "the benchmarks of your category.",
    "company": "Hi! I can answer platform-wide questions about orders, revenue, sellers, customers, deliveries "
               "and reviews.",
}

# Example questions shown as chips on an empty chat; each stays inside the role's rules in agent/policy.py
STARTER_QUESTIONS = {
    "customer": [
        "What are the 5 best-rated sellers for health & beauty?",
        "What do reviewers complain about most in bed_bath_table?",
        "Which cheap toys get the best reviews?",
        "What's the average delivery time to RJ?",
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

# Columns that identify a customer (policy S3/C3): never shown to a seller or shopper, even in a data table
HIDDEN_COLUMNS = re.compile(r"customer_(unique_)?id|zip", re.IGNORECASE)
TIME_COLUMN = re.compile(r"month|year|date|week|day", re.IGNORECASE)
MARKDOWN_TABLE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.MULTILINE)  # header separator row
MAX_TABLES = 3

@st.cache_resource
def get_supervisor(role, user_id, prompt_version=3):
    """Get the supervisor agent for the specified role."""
    return supervisor_agent.supervisor_agent(role, user_id)


def visible_turns(messages):
    """Pair each user message with only the final assistant answer for that turn."""
    turns = []
    user_message = None
    assistant_messages = []

    def add_turn():
        if user_message is not None:
            final_answer = next(
                (message for message in reversed(assistant_messages)
                 if message.text and not message.text.startswith("Transferring")),
                None,
            )
            turns.append((user_message, final_answer))

    for message in messages:
        if isinstance(message, HumanMessage):
            add_turn()
            user_message = message
            assistant_messages = []
        elif isinstance(message, AIMessage):
            assistant_messages.append(message)

    add_turn()
    return turns


def ask_chip(key):
    """Send the clicked chip as if it was typed in the chat input (same shape: text, files, audio)."""
    question = st.session_state.get(key)
    st.session_state[key] = None
    if question:
        st.session_state["pending_prompt"] = SimpleNamespace(text=question, files=[], audio=None)


def new_chat():
    """Start over without logging out."""
    st.session_state["messages"] = []
    st.session_state["failed_prompt"] = None
    st.session_state["last_error"] = None


def retry_failed_prompt():
    st.session_state["pending_prompt"] = st.session_state.get("failed_prompt")
    st.session_state["failed_prompt"] = None
    st.session_state["last_error"] = None


def chat_transcript(turns):
    """The visible conversation as Markdown, for the download button."""
    parts = []
    for user_message, assistant_message in turns:
        parts.append(f"**You:** {user_message.additional_kwargs.get('display_content', user_message.content)}")
        if assistant_message:
            parts.append(f"**Assistant:** {supervisor_agent.clean_answer(assistant_message.text)}")
        parts.append("---")
    return "\n\n".join(parts)


def activity_label(event):
    """Map graph node events to safe, user-facing progress labels."""
    namespace = "/".join(event.get("ns", ()))
    if "review_agent" in namespace:
        return "Review specialist is searching customer feedback"
    if "sql_agent" in namespace or "data_agent" in namespace:
        return "SQL specialist is analyzing Olist data"

    if event.get("type") == "updates":
        node_names = event.get("data", {}).keys()
        if "supervisor" in node_names:
            return "The agent is preparing an answer"
        if any(name.startswith("transfer_to_") for name in node_names):
            return "The agent is handing the request to a specialist"
    return None


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


def latest_versions(tables):
    """The data agent can be asked twice or retry a query: of tables with the same columns, keep only the last."""
    return [table for number, table in enumerate(tables)
            if not any(list(later.columns) == list(table.columns) for later in tables[number + 1:])]


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


def show_refusal(message, role, key, latest):
    """A refused request (agent/policy.py) as a notice, with allowed questions to try on the latest turn."""
    st.info(supervisor_agent.clean_answer(message.text), icon="🔒")
    if latest:
        st.pills("You could ask instead", STARTER_QUESTIONS.get(role, [])[:3], key=f"refusal_chip_{key}",
                 on_change=ask_chip, args=(f"refusal_chip_{key}",))


st.title("Olist Virtual Assistant 🛍️")

login_page()

# Note: Tambah observasibility

if not st.session_state["logged_in"]:
	st.stop()

role = st.session_state["role"]
id_ = st.session_state.get("id", "N/A")
busy = st.session_state.get("pending_prompt") is not None
turns = visible_turns(st.session_state["messages"])

with st.sidebar:
    st.divider()
    st.button("New chat", icon="➕", on_click=new_chat, disabled=busy or not turns, width="stretch")
    st.download_button("Download chat", chat_transcript(turns), file_name="olist_chat.md", mime="text/markdown",
                       icon="⬇️", disabled=not turns, width="stretch")

st.header(f"{role} assistant")
st.caption("Ask a question to get help with Olist.")

if not turns and not busy:
    with st.container(border=True):
        st.markdown(ROLE_INTRO.get(role, ""))
        st.pills("Try asking", STARTER_QUESTIONS.get(role, []), key="starter_chip",
                 on_change=ask_chip, args=("starter_chip",))

for number, (user_message, assistant_message) in enumerate(turns):
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_message.additional_kwargs.get("display_content", user_message.content))
        if user_message.additional_kwargs.get("image"):
            st.image(user_message.additional_kwargs["image"], width=300)
        if user_message.additional_kwargs.get("audio"):
            st.audio(user_message.additional_kwargs["audio"])
        if user_message.additional_kwargs.get("transcript"):
            st.caption(f"🎤 {user_message.additional_kwargs['transcript']}")
    if assistant_message:
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            if assistant_message.additional_kwargs.get("violated_rule"):
                show_refusal(assistant_message, role, number, latest=number == len(turns) - 1 and not busy)
            else:
                answer = supervisor_agent.clean_answer(assistant_message.text)
                st.markdown(answer)
                show_tables(assistant_message.additional_kwargs.get("tables", []), number,
                            collapsed=bool(MARKDOWN_TABLE.search(answer)))
            show_answer_details(assistant_message)

if st.session_state.get("last_error") and not busy:
    st.error(st.session_state["last_error"])
    if st.session_state.get("failed_prompt"):
        st.button("Retry", icon="🔄", on_click=retry_failed_prompt)

# An answer takes 20-40 s. A message sent meanwhile reran the script and threw away the question in progress,
# so the input is disabled while one is pending: a submitted prompt is kept in session_state, then the page reruns.
if submitted := st.chat_input("Message the Olist assistant", accept_file = True, accept_audio = True,
                              disabled = busy):
    st.session_state["pending_prompt"] = submitted
    st.rerun()

if prompt := st.session_state.get("pending_prompt"):
    st.session_state["last_error"] = None
    st.session_state["failed_prompt"] = None
    started = time.monotonic()

    # Extract the text, files, and audio from the prompt
    prompt_text = prompt.text
    uploaded_files = prompt.files
    recorded_audio = prompt.audio

    # Store additional media in kwargs to pass to LangChain agent
    additional_kwargs = {}
    if uploaded_files:
        additional_kwargs["image"] = uploaded_files[0].getvalue()
    if recorded_audio:
        additional_kwargs["audio"] = recorded_audio.getvalue()

    # Create user message with additional media and append to session state
    user_message = HumanMessage(content=prompt_text, additional_kwargs=additional_kwargs)
    input_messages = [*st.session_state["messages"], user_message]

    # Display the user's message in the chat interface
    with st.chat_message("user", avatar=AVATARS["user"]):
        if prompt_text:
            st.markdown(prompt_text)
        # Render the attachments in the chat history
        if uploaded_files:
            for file in uploaded_files:
                st.image(file, width=300)
        if recorded_audio:
            st.audio(recorded_audio)

    # Processing assistant response in a status spinner to indicate that the system is working
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.status("Working on a response...", expanded=True) as activity_status:
            activity_steps = []
            sql_results = []
            sql_agent.collected_results.set(sql_results)    # filled by the SQL tool, shown as tables
            try:
                supervisor = get_supervisor(role, st.session_state.get("agent_id"), prompt_version=3)
                spoken_text = ""
                if uploaded_files or recorded_audio:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        image_path = None
                        audio_path = None
                        if uploaded_files:
                            image_path = Path(temp_dir) / f"image{Path(uploaded_files[0].name).suffix}"
                            image_path.write_bytes(uploaded_files[0].getvalue())
                        if recorded_audio:
                            audio_path = Path(temp_dir) / f"audio{Path(recorded_audio.name).suffix}"
                            audio_path.write_bytes(recorded_audio.getvalue())

                        spoken_text = input_preprocessing.voice_to_text(audio_path) if audio_path else ""
                        if spoken_text:
                            # Show what Whisper heard, now and in the chat history (also on a refusal)
                            st.caption(f"🎤 {spoken_text}")
                            additional_kwargs["transcript"] = spoken_text
                            user_message.additional_kwargs["transcript"] = spoken_text
                        user_text = " ".join(part for part in (prompt_text, spoken_text) if part).strip()
                        question = input_preprocessing.build_question(
                            text=user_text,
                            image_path=image_path,
                        )
                else:
                    user_text = prompt_text.strip()
                    question = user_text

                recent_context = "\n".join(
                    f"{type(message).__name__}: {message.text}"
                    for message in st.session_state["messages"][-8:]
                    if isinstance(message, (HumanMessage, AIMessage)) and message.text
                )
                scope_decision = supervisor_agent.check_request_scope(
                    role=role,
                    question=question,
                    user_id=st.session_state.get("agent_id"),
                    conversation_context=recent_context,
                    original_text=user_text,
                )

                if not scope_decision.allowed:
                    # violated_rule (e.g. "C1", see agent/policy.py) is not shown, it marks the answer as a refusal
                    final_answer = AIMessage(content=scope_decision.reply,
                                             additional_kwargs={"violated_rule": scope_decision.violated_rule
                                                                or "denied"})
                    st.session_state["messages"] = [*input_messages, final_answer]
                else:
                    supervisor_message = HumanMessage(
                        content=(
                            f"{question}\n\nANSWER LANGUAGE: {scope_decision.answer_language}. "
                            f"Respond to the user and instruct every specialist to answer in "
                            f"{scope_decision.answer_language}."
                        ),
                        additional_kwargs={
                            **additional_kwargs,
                            "answer_language": scope_decision.answer_language,
                            "display_content": prompt_text,    # the transcript is shown separately
                        },
                    )
                    supervisor_input = [*st.session_state["messages"], supervisor_message]
                    final_state = {}
                    for event in supervisor.stream(
                        {"messages": supervisor_input},
                        stream_mode=["updates", "values"],
                        subgraphs=True,
                        version="v2",
                    ):
                        if event["type"] == "values" and not event.get("ns"):
                            final_state = event["data"]

                        label = activity_label(event)
                        if label and (not activity_steps or activity_steps[-1] != label):
                            activity_steps.append(label)
                            st.write(label)
                            activity_status.update(
                                label=f"Working on a response... ({time.monotonic() - started:.0f} s)")

                    updated_messages = final_state.get("messages", supervisor_input)
                    st.session_state["messages"] = updated_messages
                    new_messages = updated_messages[len(supervisor_input):]
                    final_answer = next(
                        (message for message in reversed(new_messages)
                         if isinstance(message, AIMessage) and message.text
                         and not message.text.startswith("Transferring")),
                        None,
                    )
                activity_status.update(label="Answer ready", state="complete", expanded=False)
            except Exception as error:
                # The question is not kept in the history; the Retry button sends it again
                activity_status.update(label="Couldn't complete the request", state="error")
                st.session_state["failed_prompt"] = prompt
                st.session_state["last_error"] = f"I couldn't get a response from the assistant: {error}"
                final_answer = None

        if final_answer:
            final_answer.additional_kwargs["activity"] = activity_steps
            final_answer.additional_kwargs["elapsed"] = time.monotonic() - started
            final_answer.additional_kwargs["tables"] = answer_tables(sql_results, role)

    st.session_state["pending_prompt"] = None
    st.rerun()      # enables the input again; the answer is shown from the chat history
