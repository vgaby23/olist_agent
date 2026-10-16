import streamlit as st
from src.agents import supervisor_agent

AGENTS_BY_ROLE = {
	"Customer": "customer_agent",
	"Seller": "seller_agent",
	"Company staff": "company_staff_agent",
}

st.set_page_config(page_title="Olist Virtual Assistant", page_icon="🛍️", layout="centered")

st.title("Welcome to Olist Virtual Assistant 🛍️")

supervisor = supervisor_agent()

# Login Page
st.caption("Choose an assistant, then start chatting.")

picker, _ = st.columns([2, 3])
with picker:
	role = st.selectbox(
		"Assistant",
		options=list(AGENTS_BY_ROLE),
		index=None,
		placeholder="Choose a role",
	)

if role is None:
	st.info("Choose Customer, Seller, or Company staff to begin.")
else:
	agent_id = AGENTS_BY_ROLE[role]
	st.session_state["user_role"] = role
	st.session_state["agent_id"] = agent_id

	conversations = st.session_state.setdefault("conversations", {})
	messages = conversations.setdefault(agent_id, [])

	st.divider()
	for message in messages:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	prompt = st.chat_input(f"Message the {role.lower()} assistant")
	if prompt:
		messages.append({"role": "user", "content": prompt})
		st.rerun()