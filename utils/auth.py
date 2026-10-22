import streamlit as st
import sqlite3

DB = {
	"customer": {"password": "customer123", "role": "customer"},
    "seller": {"password": "seller123", "role": "seller"}, 
    "admin" :{"password": "admin123", "role": "company"}
}

# customer: top level review -> (di supervisor)forbidden actions: 

def verify_login():
    account = st.session_state.get("login_account")
    password = st.session_state.get("password")

    if account not in DB or DB[account]["password"] != password:
        st.error("Invalid username or password. Please try again.")
        return

    role = DB[account]["role"]
    user_id = (st.session_state.get("login_user_id") or "").strip().lower()
    if role in {"seller", "customer"}:
        if not user_id:
            st.error(f"Enter your {role} ID to continue.")
            return
        try:
            from dataset.preprocessing.numeric_data.sql_tools import make_scope

            scope = make_scope(role, user_id)
        except (ValueError, OSError, sqlite3.Error) as error:
            st.error(f"Could not verify {role} ID: {error}")
            return
        user_id = scope.user_id

    st.session_state["logged_in"] = True
    st.session_state["username"] = account
    st.session_state["role"] = role
    st.session_state["agent_id"] = user_id if role in {"seller", "customer"} else None
    st.session_state["login_user_id"] = ""
    st.session_state["password"] = ""

def logout():
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.session_state["role"] = None
    st.session_state["agent_id"] = None
    st.session_state["login_user_id"] = ""
    st.session_state['password'] = ""
    st.session_state['conversations'] = {}
    st.session_state['messages'] = []

def login_page():
    if not st.session_state.get("logged_in"):
        st.title("Platform Login")

        users = list(DB.keys())
        account = st.selectbox("Role", options=users, index=0, key="login_account")
        role = DB[account]["role"]
        if role in {"seller", "customer"}:
            st.text_input(f"{role.title()} ID", key="login_user_id")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=verify_login)
    else:
        with st.sidebar:
            st.write(f"Logged in as: **{st.session_state['username']}**")
            if st.session_state.get("agent_id"):
                st.caption(f"ID: {st.session_state['agent_id']}")
            st.write(f"Role: *{st.session_state['role']}*")
            st.button("Logout", on_click=logout)