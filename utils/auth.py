import streamlit as st
from utils.api_client import APIRequestError, validate_seller_id

DB = {
	"customer": {"password": "customer123", "role": "customer"},
    "seller": {"password": "seller123", "role": "seller"}, 
    "company" :{"password": "company123", "role": "company"}
}

def verify_login():
    account = st.session_state.get("login_account")
    password = st.session_state.get("password")

    if account not in DB or DB[account]["password"] != password:
        st.error("Invalid username or password. Please try again.")
        return

    role = DB[account]["role"]
    user_id = (st.session_state.get("login_user_id") or "").strip().lower()
    if role == "seller":
        if not user_id:
            st.error(f"Enter your {role} ID to continue.")
            return
        try:
            user_id = validate_seller_id(user_id)
        except APIRequestError as error:
            st.error(f"Could not verify {role} ID: {error}")
            return

    st.session_state["logged_in"] = True
    st.session_state["username"] = account
    st.session_state["role"] = role
    st.session_state["agent_id"] = user_id if role == "seller" else None
    st.session_state["messages"] = []
    st.session_state["active_chat_id"] = None
    st.session_state["pending_prompt"] = None
    st.session_state["failed_prompt"] = None
    st.session_state["last_error"] = None
    st.session_state["login_user_id"] = ""
    st.session_state["password"] = ""

def logout():
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.session_state["role"] = None
    st.session_state["agent_id"] = None
    st.session_state["login_user_id"] = ""
    st.session_state['password'] = ""
    st.session_state['messages'] = []
    st.session_state['pending_prompt'] = None
    st.session_state['active_chat_id'] = None
    st.session_state['failed_prompt'] = None
    st.session_state['last_error'] = None

ACCOUNT_LABELS = {"customer": "🛒 Customer", "seller": "🏪 Seller", "company": "🏢 Company"}
ACCOUNT_HELP = {
    "customer": "Find products and sellers and read what other shoppers say. No account ID needed.",
    "seller": "Your own shop's sales, deliveries and reviews, plus category benchmarks.",
    "company": "Platform-wide data on orders, revenue, sellers, customers and reviews.",
}

# Test seller for the demo: the seller with the most orders (1,854) and reviews (1,835) in the anonymised dataset
SAMPLE_SELLER_ID = "6560211a19b47992c3666cc44a7e94c0"

def use_sample_seller():
    st.session_state["login_user_id"] = SAMPLE_SELLER_ID

def login_page(on_logout=logout):
    if not st.session_state.get("logged_in"):
        st.caption("Your assistant for shopping on Olist and for running your shop.")

        with st.container(border=True):
            st.subheader("Log in")
            users = list(DB.keys())
            account = st.selectbox("I am a", options=users, index=0, key="login_account",
                                   format_func=lambda name: ACCOUNT_LABELS.get(name, name))
            st.caption(ACCOUNT_HELP.get(account, ""))
            role = DB[account]["role"]
            if role == "seller":
                st.text_input("Seller ID", key="login_user_id", placeholder="32-character seller ID")
                st.caption(f"For testing: `{SAMPLE_SELLER_ID}`, the seller with the most orders and reviews.")
                st.button("Use sample seller ID", icon="🧪", on_click=use_sample_seller)

            st.text_input("Password", type="password", key="password")
            st.caption(f'Password format: role(lowercase) + 123, e.g. "customer123" for a customer.')

            st.button("Log in", on_click=verify_login, type="primary", width="stretch")
    else:
        with st.sidebar:
            st.write(f"Logged in as: **{st.session_state['username']}**")
            if st.session_state.get("agent_id"):
                st.caption(f"ID: {st.session_state['agent_id']}")
            st.button("Logout", on_click=on_logout)