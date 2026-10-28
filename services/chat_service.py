import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent import sql_agent, supervisor_agent
from utils import input_preprocessing
from utils.chat_results import answer_tables, collect_token_usage
from utils.chat_runtime import activity_label, handoff_call_ids, message_flow_entries


@lru_cache(maxsize=20)
def get_supervisor(role: str, user_id: str | None):
    return supervisor_agent.supervisor_agent(role, user_id)


def process_chat(
    *,
    role: str,
    user_id: str | None,
    message: str,
    history: Sequence[BaseMessage] = (),
    image: bytes | None = None,
    image_suffix: str | None = None,
    audio: bytes | None = None,
    audio_suffix: str | None = None,
) -> dict:
    """Run one role-scoped assistant turn outside the Streamlit interface."""
    started = time.monotonic()

    activity_steps = []
    message_flow = []
    sql_results = []
    collected_results_token = sql_agent.collected_results.set(sql_results)
    try:
        spoken_text = ""
        if image is not None or audio is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                image_path = None
                audio_path = None
                if image is not None:
                    image_path = Path(temp_dir) / f"image{image_suffix or '.jpg'}"
                    image_path.write_bytes(image)
                if audio is not None:
                    audio_path = Path(temp_dir) / f"audio{audio_suffix or '.wav'}"
                    audio_path.write_bytes(audio)
                spoken_text = (
                    input_preprocessing.voice_to_text(audio_path) if audio_path else ""
                )
                user_text = " ".join(
                    part for part in (message, spoken_text) if part
                ).strip()
                question = input_preprocessing.build_question(
                    text=user_text,
                    image_path=image_path,
                )
        else:
            user_text = message.strip()
            question = user_text

        recent_context = "\n".join(
            f"{type(item).__name__}: {item.text}"
            for item in history[-8:]
            if isinstance(item, (HumanMessage, AIMessage)) and item.text
        )
        scope_decision = supervisor_agent.check_request_scope(
            role=role,
            question=question,
            user_id=user_id,
            conversation_context=recent_context,
            original_text=user_text,
        )
        if not scope_decision.allowed:
            final_answer = AIMessage(
                content=scope_decision.reply,
                additional_kwargs={
                    "violated_rule": scope_decision.violated_rule or "denied"
                },
            )
            updated_messages = [*history, HumanMessage(content=message), final_answer]
            token_usage = None
        else:
            message_details: dict[str, str | bytes] = {
                "answer_language": scope_decision.answer_language,
                "display_content": message,
            }
            if image is not None:
                message_details["image"] = image
            if audio is not None:
                message_details["audio"] = audio
            if spoken_text:
                message_details["transcript"] = spoken_text

            supervisor_message = HumanMessage(
                content=(
                    f"{question}\n\nANSWER LANGUAGE: {scope_decision.answer_language}. "
                    f"Respond to the user and instruct every specialist to answer in "
                    f"{scope_decision.answer_language}."
                ),
                additional_kwargs=message_details,
            )
            supervisor_input = [*history, supervisor_message]
            previous_handoffs = handoff_call_ids(supervisor_input)
            seen_handoffs = set(previous_handoffs)
            final_state = {}
            supervisor = get_supervisor(role, user_id)

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
                for entry in message_flow_entries(event, previous_handoffs):
                    call_id = entry.get("call_id")
                    if call_id and call_id in seen_handoffs:
                        continue
                    if call_id:
                        seen_handoffs.add(call_id)
                    message_flow.append(
                        {key: value for key, value in entry.items() if key != "call_id"}
                    )

            updated_messages = final_state.get("messages", supervisor_input)
            new_messages = updated_messages[len(supervisor_input):]
            token_usage = collect_token_usage(new_messages)
            final_answer = next(
                (
                    item
                    for item in reversed(new_messages)
                    if isinstance(item, AIMessage)
                    and item.text
                    and not item.text.startswith("Transferring")
                ),
                None,
            )

        tables = answer_tables(sql_results, role)
        elapsed = time.monotonic() - started
        if final_answer:
            final_answer.additional_kwargs["activity"] = activity_steps
            final_answer.additional_kwargs["message_flow"] = message_flow
            final_answer.additional_kwargs["elapsed"] = elapsed
            final_answer.additional_kwargs["tables"] = tables
            if spoken_text:
                final_answer.additional_kwargs["transcript"] = spoken_text
            if token_usage:
                final_answer.additional_kwargs["token_usage"] = token_usage

        return {
            "answer": final_answer,
            "messages": updated_messages,
            "activity": activity_steps,
            "message_flow": message_flow,
            "elapsed_seconds": elapsed,
            "token_usage": token_usage,
            "tables": tables,
            "transcript": spoken_text,
        }
    finally:
        sql_agent.collected_results.reset(collected_results_token)
