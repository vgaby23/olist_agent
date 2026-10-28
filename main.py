import logging
import time
import streamlit as st
import tempfile
from pathlib import Path

from agent import supervisor_agent, sql_agent
from utils.auth import login_page
from utils import input_preprocessing
from utils.chat_config import MARKDOWN_TABLE, STARTER_QUESTIONS
from utils.chat_display import (
    answer_tables,
    collect_token_usage,
    markdown_safe,
    show_answer_details,
    show_refusal,
    show_tables,
    show_message_flow,
    show_token_usage,
)
from utils.chat_history import (
    archive_active_conversation,
    ask_chip,
    chat_transcript,
    conversation_owner,
    logout_and_archive,
    new_chat,
    retry_failed_prompt,
    select_conversation,
    visible_turns,
)
from utils.chat_runtime import activity_label, handoff_call_ids, message_flow_entries
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
st.session_state.setdefault("conversations", {})
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

@st.cache_resource(max_entries=20, ttl="1h")
def get_supervisor(role, user_id, prompt_version=3):
    """Get the supervisor agent for the specified role."""
    return supervisor_agent.supervisor_agent(role, user_id)


st.title("Olist Virtual Assistant 🛍️")

login_page(on_logout=logout_and_archive)

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
    st.download_button("Download chat", chat_transcript(turns, supervisor_agent), file_name="olist_chat.md", mime="text/markdown",
                       icon="⬇️", disabled=not turns, width="stretch")
    owner = conversation_owner()
    saved_conversations = st.session_state["conversations"].get(owner, []) if owner else []
    if saved_conversations:
        st.subheader("Conversation history")
        for conversation in saved_conversations:
            is_active = conversation["id"] == st.session_state.get("active_chat_id")
            st.button(
                conversation["title"],
                key=f"conversation_{conversation['id']}",
                type="primary" if is_active else "secondary",
                disabled=busy,
                width="stretch",
                on_click=select_conversation,
                args=(owner, conversation["id"]),
            )

st.header(f"{role.capitalize()} assistant")
st.caption("Ask a question to get help with Olist.")

if not turns and not busy:
    with st.container(border=True):
        st.markdown(ROLE_INTRO.get(role, ""))
        st.pills("Try asking", STARTER_QUESTIONS.get(role, []), key="starter_chip",
                 on_change=ask_chip, args=("starter_chip",))

for number, (user_message, assistant_message) in enumerate(turns):
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(markdown_safe(user_message.additional_kwargs.get("display_content", user_message.content)))
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
                st.markdown(markdown_safe(answer))
                show_tables(assistant_message.additional_kwargs.get("tables", []), number,
                            collapsed=bool(MARKDOWN_TABLE.search(answer)))
            show_answer_details(assistant_message)
            show_message_flow(assistant_message.additional_kwargs.get("message_flow", []))
            show_token_usage(assistant_message)

if st.session_state.get("last_error") and not busy:
    st.error(st.session_state["last_error"])
    if st.session_state.get("failed_prompt"):
        st.button("Retry", icon="🔄", on_click=retry_failed_prompt)

# An answer takes 20-40 s. A message sent meanwhile reran the script and threw away the question in progress,
# so the input is disabled while one is pending: a submitted prompt is kept in session_state, then the page reruns.
if submitted := st.chat_input("Message the Olist assistant", accept_file = True, accept_audio = True, file_type=["png", "jpg", "jpeg", "webp"],
                              disabled = busy):
    st.session_state["pending_prompt"] = submitted
    st.rerun()

if prompt := st.session_state.get("pending_prompt"):
    # Cleared right away: if anything below crashes, the next rerun must not run the same prompt again
    # with the input still disabled
    st.session_state["pending_prompt"] = None
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
            st.markdown(markdown_safe(prompt_text))
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
            message_flow = []
            sql_results = []
            token_usage = None
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
                    previous_handoffs = handoff_call_ids(supervisor_input)
                    seen_handoffs = set(previous_handoffs)
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
                        for entry in message_flow_entries(event, previous_handoffs):
                            call_id = entry.get("call_id")
                            if call_id and call_id in seen_handoffs:
                                continue
                            if call_id:
                                seen_handoffs.add(call_id)
                            visible_entry = {
                                key: value for key, value in entry.items() if key != "call_id"
                            }
                            message_flow.append(visible_entry)
                            st.markdown(
                                f"**{visible_entry['sender']} -> {visible_entry['recipient']}**"
                            )
                            st.code(visible_entry["message"], language=None)

                    updated_messages = final_state.get("messages", supervisor_input)
                    st.session_state["messages"] = updated_messages
                    new_messages = updated_messages[len(supervisor_input):]
                    token_usage = collect_token_usage(new_messages)
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
            final_answer.additional_kwargs["message_flow"] = message_flow
            final_answer.additional_kwargs["elapsed"] = time.monotonic() - started
            try:
                final_answer.additional_kwargs["tables"] = answer_tables(sql_results, role)
                if token_usage:
                    final_answer.additional_kwargs["token_usage"] = token_usage
                archive_active_conversation()
            except Exception as error:      # the answer itself is already in the history; only extras failed
                st.session_state["last_error"] = f"The answer is shown, but its tables or history could not be saved: {error}"

    st.rerun()      # enables the input again; the answer is shown from the chat history
