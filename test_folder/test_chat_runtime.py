from types import SimpleNamespace

from utils.chat_runtime import handoff_call_ids, message_flow_entries


def test_extracts_supervisor_handoff_task():
    message = SimpleNamespace(
        tool_calls=[{
            "name": "transfer_to_sql_agent",
            "args": {"task": "Count orders for this month."},
            "id": "handoff-1",
        }]
    )
    event = {
        "type": "updates",
        "ns": [],
        "data": {"supervisor": {"messages": [message]}},
    }

    assert message_flow_entries(event) == [{
        "sender": "Supervisor",
        "recipient": "SQL specialist",
        "message": "Count orders for this month.",
        "call_id": "handoff-1",
    }]


def test_ignores_handoff_calls_repeated_outside_supervisor_update():
    message = SimpleNamespace(
        tool_calls=[{
            "name": "transfer_to_sql_agent",
            "args": {"task": "Count orders for this month."},
            "id": "handoff-1",
        }]
    )
    event = {
        "type": "updates",
        "ns": [],
        "data": {
            "supervisor": {"messages": [message]},
            "transfer_to_sql_agent": {"messages": [message]},
        },
    }

    assert message_flow_entries(event) == [{
        "sender": "Supervisor",
        "recipient": "SQL specialist",
        "message": "Count orders for this month.",
        "call_id": "handoff-1",
    }]


def test_excludes_handoffs_from_prior_conversation_turns():
    old_handoff = SimpleNamespace(
        tool_calls=[{
            "name": "transfer_to_sql_agent",
            "args": {"task": "Old request."},
            "id": "old-handoff",
        }]
    )
    current_handoff = SimpleNamespace(
        tool_calls=[{
            "name": "transfer_to_sql_agent",
            "args": {"task": "Current request."},
            "id": "current-handoff",
        }]
    )
    previous_handoffs = handoff_call_ids([old_handoff])
    event = {
        "type": "updates",
        "ns": [],
        "data": {"supervisor": {"messages": [old_handoff, current_handoff]}},
    }

    assert message_flow_entries(event, previous_handoffs) == [{
        "sender": "Supervisor",
        "recipient": "SQL specialist",
        "message": "Current request.",
        "call_id": "current-handoff",
    }]


def test_extracts_only_specialist_reply_returned_to_supervisor():
    intermediate = SimpleNamespace(type="ai", text="Intermediate value.", tool_calls=[])
    reply = SimpleNamespace(type="ai", text="There were 120 orders.", tool_calls=[])
    tool_result = SimpleNamespace(type="tool", text="internal result")
    event = {
        "type": "updates",
        "ns": [],
        "data": {"sql_agent": {"messages": [intermediate, tool_result, reply]}},
    }

    assert message_flow_entries(event) == [{
        "sender": "SQL specialist",
        "recipient": "Supervisor",
        "message": "There were 120 orders.",
    }]


def test_ignores_nested_specialist_updates():
    reply = SimpleNamespace(type="ai", text="Intermediate value.", tool_calls=[])
    event = {
        "type": "updates",
        "ns": ["sql_agent:run-1"],
        "data": {"agent": {"messages": [reply]}},
    }

    assert message_flow_entries(event) == []


def test_ignores_non_update_events_and_unknown_handoffs():
    message = SimpleNamespace(
        tool_calls=[{
            "name": "transfer_to_unknown_agent",
            "args": {"task": "Do something."},
        }]
    )

    assert message_flow_entries({"type": "values", "ns": [], "data": {}}) == []
    assert message_flow_entries({
        "type": "updates",
        "ns": [],
        "data": {"supervisor": {"messages": [message]}},
    }) == []
