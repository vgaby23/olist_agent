import logging
import streamlit as st

from utils.auth import login_page
from utils.chat_history import logout_and_archive, visible_turns
from interface import setup_page, render_chat_history, sidebar, handle_pending_prompt

# Print our own INFO logs (e.g. the voice transcript) in the terminal running streamlit
logging.basicConfig(format="%(asctime)s %(name)s: %(message)s")
logging.getLogger("utils").setLevel(logging.INFO)

setup_page()

st.title("Olist Virtual Assistant 🛍️")

login_page(on_logout=logout_and_archive)

if not st.session_state["logged_in"]:
    st.stop()

role = st.session_state["role"]
busy = st.session_state.get("pending_prompt") is not None
turns = visible_turns(st.session_state["messages"])

sidebar(busy, turns)
render_chat_history(role, turns, busy)

# An answer takes 20-40 s; keep the submitted prompt while the API request is running.
if submitted := st.chat_input(
    "Message the Olist assistant",
    accept_file=True,
    accept_audio=True,
    file_type=["png", "jpg", "jpeg", "webp"],
    disabled=busy,
):
    st.session_state["pending_prompt"] = submitted
    st.rerun()

if prompt := st.session_state.get("pending_prompt"):
    handle_pending_prompt(prompt, role)