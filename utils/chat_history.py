import streamlit as st
import uuid
import time
from langchain_core.messages import HumanMessage, AIMessage
from types import SimpleNamespace
from utils.auth import logout


MAX_SAVED_CONVERSATIONS = 20

def conversation_owner():
    """Keep chat history separate for each account and seller ID."""
    username = st.session_state.get("username")
    if not username:
        return None
    return f"{username}:{st.session_state.get('agent_id') or ''}"

def retry_failed_prompt():
    st.session_state["pending_prompt"] = st.session_state.get("failed_prompt")
    st.session_state["failed_prompt"] = None
    st.session_state["last_error"] = None

def archive_active_conversation():
    """Save the active chat to this account's bounded sidebar history."""
    messages = st.session_state.get("messages", [])
    owner = conversation_owner()
    if not owner or not messages:
        return

    conversations = st.session_state.setdefault("conversations", {})
    owner_conversations = conversations.setdefault(owner, [])
    chat_id = st.session_state.get("active_chat_id") or uuid.uuid4().hex
    first_user_message = next(
        (message for message in messages if isinstance(message, HumanMessage)),
        None,
    )
    title = ""
    if first_user_message:
        title = first_user_message.additional_kwargs.get(
            "display_content", first_user_message.content
        )
    if not isinstance(title, str):
        title = ""
    title = " ".join(title.split())[:60] or "Image/audio conversation"

    conversation = next(
        (item for item in owner_conversations if item["id"] == chat_id),
        None,
    )
    if conversation is None:
        conversation = {"id": chat_id}
        owner_conversations.append(conversation)
    conversation.update({"title": title, "messages": list(messages), "updated_at": time.time()})
    owner_conversations.sort(key=lambda item: item["updated_at"], reverse=True)
    del owner_conversations[MAX_SAVED_CONVERSATIONS:]
    st.session_state["active_chat_id"] = chat_id

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

def select_conversation(owner, chat_id):
    """Switch to an archived chat after preserving the current one."""
    archive_active_conversation()
    conversation = next(
        (item for item in st.session_state.get("conversations", {}).get(owner, [])
         if item["id"] == chat_id),
        None,
    )
    if conversation:
        st.session_state["active_chat_id"] = chat_id
        st.session_state["messages"] = list(conversation["messages"])
        st.session_state["failed_prompt"] = None
        st.session_state["last_error"] = None

def logout_and_archive():
    archive_active_conversation()
    logout()


def new_chat():
    """Archive the current chat and start a blank one."""
    archive_active_conversation()
    st.session_state["active_chat_id"] = None
    st.session_state["messages"] = []
    st.session_state["failed_prompt"] = None
    st.session_state["last_error"] = None

def ask_chip(key):
    """Send the clicked chip as if it was typed in the chat input (same shape: text, files, audio)."""
    question = st.session_state.get(key)
    st.session_state[key] = None
    if question:
        st.session_state["pending_prompt"] = SimpleNamespace(text=question, files=[], audio=None)


def chat_transcript(turns, supervisor_agent):
    """The visible conversation as Markdown, for the download button."""
    parts = []
    for user_message, assistant_message in turns:
        parts.append(f"**You:** {user_message.additional_kwargs.get('display_content', user_message.content)}")
        if assistant_message:
            parts.append(f"**Assistant:** {supervisor_agent.clean_answer(assistant_message.text)}")
        parts.append("---")
    return "\n\n".join(parts)
