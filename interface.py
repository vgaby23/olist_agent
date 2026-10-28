import logging
import time

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from utils.api_client import APIRequestError, send_chat
from utils.chat_config import MARKDOWN_TABLE, STARTER_QUESTIONS
from utils.chat_display import (
    markdown_safe,
    show_answer_details,
    show_message_flow,
    show_refusal,
    show_tables,
    show_token_usage,
)
from utils.chat_history import (
    archive_active_conversation,
    ask_chip,
    chat_transcript,
    conversation_owner,
    new_chat,
    retry_failed_prompt,
    select_conversation,
    visible_turns,
)
from utils.chat_text import clean_answer


log = logging.getLogger(__name__)
WIDE_ROLES = {"company", "seller"}
AVATARS = {"user": "🧑", "assistant": "🛍️"}

ROLE_INTRO = {
    "customer": "Hi! I can help you find well-rated products and sellers, tell you what other shoppers say in "
    "their reviews, and how long deliveries usually take to your state.",
    "seller": "Hi! I can analyse your shop's sales, revenue, deliveries and reviews, and compare your shop with "
    "the benchmarks of your category.",
    "company": "Hi! I can answer platform-wide questions about orders, revenue, sellers, customers, deliveries "
    "and reviews.",
}


def setup_page():
    """Configure the Streamlit interface page."""
    st.set_page_config(
        page_title="Olist Virtual Assistant",
        page_icon="🛍️",
        layout="wide" if st.session_state.get("role") in WIDE_ROLES else "centered",
    )
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("role", None)
    st.session_state.setdefault("conversations", {})
    st.session_state.setdefault("messages", [])


def sidebar(busy: bool, turns: list):
    with st.sidebar:
        st.divider()
        st.button(
            "New chat",
            icon="➕",
            on_click=new_chat,
            disabled=busy or not turns,
            width="stretch",
        )
        st.download_button(
            "Download chat",
            chat_transcript(turns),
            file_name="olist_chat.md",
            mime="text/markdown",
            icon="⬇️",
            disabled=not turns,
            width="stretch",
        )
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


def render_chat_history(role: str, turns: list, busy: bool):
    st.header(f"{role.capitalize()} assistant")
    st.caption("Ask a question to get help with Olist.")

    if not turns and not busy:
        with st.container(border=True):
            st.markdown(ROLE_INTRO.get(role, ""))
            st.pills(
                "Try asking",
                STARTER_QUESTIONS.get(role, []),
                key="starter_chip",
                on_change=ask_chip,
                args=("starter_chip",),
            )

    for number, (user_message, assistant_message) in enumerate(turns):
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(
                markdown_safe(
                    user_message.additional_kwargs.get("display_content", user_message.content)
                )
            )
            if user_message.additional_kwargs.get("image"):
                st.image(user_message.additional_kwargs["image"], width=300)
            if user_message.additional_kwargs.get("audio"):
                st.audio(user_message.additional_kwargs["audio"])
            if user_message.additional_kwargs.get("transcript"):
                st.caption(f"🎤 {user_message.additional_kwargs['transcript']}")
        if assistant_message:
            with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                if assistant_message.additional_kwargs.get("violated_rule"):
                    show_refusal(
                        assistant_message,
                        role,
                        number,
                        latest=number == len(turns) - 1 and not busy,
                    )
                else:
                    answer = clean_answer(assistant_message.text)
                    st.markdown(markdown_safe(answer))
                    show_tables(
                        assistant_message.additional_kwargs.get("tables", []),
                        number,
                        collapsed=bool(MARKDOWN_TABLE.search(answer)),
                    )
                show_answer_details(assistant_message)
                show_message_flow(
                    assistant_message.additional_kwargs.get("message_flow", [])
                )
                show_token_usage(assistant_message)

    if st.session_state.get("last_error") and not busy:
        st.error(st.session_state["last_error"])
        if st.session_state.get("failed_prompt"):
            st.button("Retry", icon="🔄", on_click=retry_failed_prompt)


def _api_history(messages):
    history = []
    for user_message, assistant_message in visible_turns(messages):
        if assistant_message is None:
            continue

        display_content = user_message.additional_kwargs.get(
            "display_content", user_message.content
        )
        transcript = user_message.additional_kwargs.get("transcript")
        image_context = user_message.additional_kwargs.get("image_context")

        parts = []
        if image_context:
            parts.append(image_context)
        if display_content:
            parts.append(display_content)
        if transcript:
            parts.append(transcript)

        user_content = " ".join(parts).strip()

        history.extend(
            [
                {"role": "user", "content": user_content or "Image/audio question"},
                {"role": "assistant", "content": clean_answer(assistant_message.text)},
            ]
        )
    return history[-20:]




def handle_pending_prompt(prompt, role: str):
    st.session_state["pending_prompt"] = None
    st.session_state["last_error"] = None
    st.session_state["failed_prompt"] = None
    started = time.monotonic()

    prompt_text = prompt.text
    uploaded_files = prompt.files
    recorded_audio = prompt.audio
    image = uploaded_files[0] if uploaded_files else None
    audio = recorded_audio

    with st.chat_message("user", avatar=AVATARS["user"]):
        if prompt_text:
            st.markdown(markdown_safe(prompt_text))
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, width=300)
        if recorded_audio:
            st.audio(recorded_audio)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.status("Working on a response...", expanded=True) as activity_status:
            try:
                result = send_chat(
                    message=prompt_text,
                    role=role,
                    user_id=st.session_state.get("agent_id"),
                    history=_api_history(st.session_state["messages"]),
                    image=(image.name, image.getvalue()) if image else None,
                    audio=(audio.name, audio.getvalue()) if audio else None,
                )
                activity_status.update(
                    label=f"Answer ready ({time.monotonic() - started:.0f} s)",
                    state="complete",
                    expanded=False,
                )
            except APIRequestError as error:
                activity_status.update(label="Couldn't complete the request", state="error")
                st.session_state["failed_prompt"] = prompt
                st.session_state["last_error"] = str(error)
                result = None

        if result:
            user_kwargs = {"display_content": prompt_text}
            if image:
                user_kwargs["image"] = image.getvalue()
            if audio:
                user_kwargs["audio"] = audio.getvalue()
            if result.get("transcript"):
                user_kwargs["transcript"] = result["transcript"]
            if result.get("image_context"):
                user_kwargs["image_context"] = result["image_context"]
            user_message = HumanMessage(content=prompt_text, additional_kwargs=user_kwargs)

            tables = [
                pd.DataFrame(table["rows"], columns=table["columns"])
                for table in result.get("tables", [])
            ]
            answer_kwargs = {
                "activity": result.get("activity", []),
                "message_flow": result.get("message_flow", []),
                "elapsed": result.get("elapsed_seconds"),
                "tables": tables,
            }
            if result.get("token_usage"):
                answer_kwargs["token_usage"] = result["token_usage"]
            if result.get("refused"):
                answer_kwargs["violated_rule"] = "denied"
            answer_message = AIMessage(
                content=result["answer"],
                additional_kwargs=answer_kwargs,
            )
            st.session_state["messages"].extend([user_message, answer_message])
            try:
                archive_active_conversation()
            except Exception as error:
                log.exception("Could not save the completed chat to Streamlit session history.")
                st.session_state["last_error"] = (
                    f"The answer is shown, but its chat history could not be saved: {error}"
                )

    st.rerun()
