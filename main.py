import streamlit as st
import tempfile
from pathlib import Path
from agent import supervisor_agent
from utils.auth import login_page
from utils import input_preprocessing
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Olist Virtual Assistant", page_icon="🛍️", layout="centered")

st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("role", None)
st.session_state.setdefault("username", None)
st.session_state.setdefault("messages", [])

@st.cache_resource
def get_supervisor(role, user_id, prompt_version=2):
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
                 if message.content and not message.content.startswith("Transferring")),
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


st.title("Olist Virtual Assistant 🛍️")

login_page()

# Note: Tambah observasibility

if not st.session_state["logged_in"]:
	st.stop()

role = st.session_state["role"]
id_ = st.session_state.get("id", "N/A")

st.header(f"{role} assistant")
st.caption("Ask a question to get help with Olist.")

for user_message, assistant_message in visible_turns(st.session_state["messages"]):
    with st.chat_message("user"):
        st.markdown(user_message.additional_kwargs.get("display_content", user_message.content))
        if user_message.additional_kwargs.get("image"):
            st.image(user_message.additional_kwargs["image"], width=300)
        if user_message.additional_kwargs.get("audio"):
            st.audio(user_message.additional_kwargs["audio"])
    if assistant_message:
        with st.chat_message("assistant"):
            st.markdown(assistant_message.content)

if prompt := st.chat_input("Message the Olist assistant", accept_file = True, accept_audio = True):

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
    with st.chat_message("user"):
        if prompt_text:
            st.markdown(prompt_text)
        # Render the attachments in the chat history
        if uploaded_files:
            for file in uploaded_files:
                st.image(file, width=300)
        if recorded_audio:
            st.audio(recorded_audio)

    # Processing assistant response in a status spinner to indicate that the system is working
    with st.chat_message("assistant"):
        with st.status("Working on a response...", expanded=True) as activity_status:
            activity_steps = []
            try:
                supervisor = get_supervisor(role, st.session_state.get("agent_id"), prompt_version=2)
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
                        user_text = " ".join(part for part in (prompt_text, spoken_text) if part).strip()
                        question = input_preprocessing.build_question(
                            text=user_text,
                            image_path=image_path,
                        )
                else:
                    user_text = prompt_text.strip()
                    question = user_text

                recent_context = "\n".join(
                    f"{type(message).__name__}: {message.content}"
                    for message in st.session_state["messages"][-8:]
                    if isinstance(message, (HumanMessage, AIMessage)) and message.content
                )
                scope_decision = supervisor_agent.check_request_scope(
                    role=role,
                    question=question,
                    user_id=st.session_state.get("agent_id"),
                    conversation_context=recent_context,
                    original_text=user_text,
                )

                if not scope_decision.allowed:
                    final_answer = AIMessage(content=scope_decision.reply)
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
                            "display_content": user_text,
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

                    updated_messages = final_state.get("messages", supervisor_input)
                    st.session_state["messages"] = updated_messages
                    new_messages = updated_messages[len(supervisor_input):]
                    final_answer = next(
                        (message for message in reversed(new_messages)
                         if isinstance(message, AIMessage) and message.content
                         and not message.content.startswith("Transferring")),
                        None,
                    )
                activity_status.update(label="Answer ready", state="complete")
            except Exception as error:
                st.session_state["messages"] = input_messages
                activity_status.update(label="Couldn't complete the request", state="error")
                st.error(f"I couldn't get a response from the assistant: {error}")
                final_answer = None

        if final_answer:
            final_answer.additional_kwargs["activity"] = activity_steps
            st.markdown(final_answer.content)