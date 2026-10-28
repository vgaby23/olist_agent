
AGENT_LABELS = {
    "sql_agent": "SQL specialist",
    "review_agent": "Review specialist",
}


def _messages_from_update(data):
    """Yield LangChain messages from a LangGraph updates payload."""
    if isinstance(data, dict):
        messages = data.get("messages", [])
        if isinstance(messages, (list, tuple)):
            yield from messages
        for key, value in data.items():
            if key != "messages":
                yield from _messages_from_update(value)
    elif isinstance(data, (list, tuple)):
        for value in data:
            yield from _messages_from_update(value)


def handoff_call_ids(messages):
    """Collect specialist handoff IDs already present in conversation history."""
    return {
        call["id"]
        for message in messages
        for call in (getattr(message, "tool_calls", None) or [])
        if call.get("name") in {f"transfer_to_{name}" for name in AGENT_LABELS}
        and call.get("id")
    }


def message_flow_entries(event, excluded_call_ids=()):
    """Extract handoffs and the single specialist reply returned to the supervisor."""
    if event.get("type") != "updates":
        return []

    namespace = event.get("ns") or ()
    if isinstance(namespace, str):
        namespace = (namespace,)
    namespace_parts = [part.split(":", 1)[0] for part in namespace]
    if namespace_parts:
        return []

    entries = []
    updates = event.get("data", {})
    if not isinstance(updates, dict):
        return entries
    for specialist, label in AGENT_LABELS.items():
        if specialist not in updates:
            continue
        messages = list(_messages_from_update(updates[specialist]))
        reply = next(
            (
                message for message in reversed(messages)
                if getattr(message, "type", None) == "ai"
                and not getattr(message, "tool_calls", None)
                and getattr(message, "text", "")
            ),
            None,
        )
        if reply:
            entries.append({
                "sender": label,
                "recipient": "Supervisor",
                "message": reply.text,
            })

    for message in _messages_from_update(updates.get("supervisor", {})):
        for call in getattr(message, "tool_calls", None) or []:
            tool_name = call.get("name", "")
            specialist = tool_name.removeprefix("transfer_to_")
            task = (call.get("args") or {}).get("task")
            if specialist in AGENT_LABELS and isinstance(task, str) and task:
                call_id = call.get("id")
                if call_id and call_id in excluded_call_ids:
                    continue
                entry = {
                    "sender": "Supervisor",
                    "recipient": AGENT_LABELS[specialist],
                    "message": task,
                }
                if call_id:
                    entry["call_id"] = call_id
                entries.append(entry)
    return entries

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