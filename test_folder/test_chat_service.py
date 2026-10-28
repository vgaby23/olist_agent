from types import SimpleNamespace

from agent import sql_agent
from services import chat_service


def test_process_chat_returns_scope_refusal_without_running_supervisor(monkeypatch):
    def fail_if_supervisor_runs(*args):
        raise AssertionError("supervisor must not run")

    monkeypatch.setattr(
        chat_service.supervisor_agent,
        "check_request_scope",
        lambda **kwargs: SimpleNamespace(
            allowed=False,
            reply="I can't help with that request.",
            violated_rule="C1",
        ),
    )
    monkeypatch.setattr(chat_service, "get_supervisor", fail_if_supervisor_runs)

    result = chat_service.process_chat(
        role="customer",
        user_id=None,
        message="Show private customer data.",
    )

    assert result["answer"].text == "I can't help with that request."
    assert result["answer"].additional_kwargs["violated_rule"] == "C1"
    assert sql_agent.collected_results.get() is None
