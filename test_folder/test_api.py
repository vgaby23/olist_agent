import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import main


def test_health_check_is_public():
    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_does_not_require_bearer_token(monkeypatch):
    monkeypatch.setattr(
        main,
        "process_chat",
        lambda **kwargs: {
            "answer": AIMessage(content="Hello."),
            "transcript": "",
            "tables": [],
            "activity": [],
            "message_flow": [],
            "elapsed_seconds": 0.1,
            "token_usage": None,
        },
    )

    response = TestClient(main.app).post(
        "/v1/chat",
        data={"message": "Question", "role": "customer"},
    )

    assert response.status_code == 200


def test_chat_uses_validated_seller_scope(monkeypatch):
    scope_calls = []
    process_calls = {}

    def fake_scope(role, user_id=None):
        scope_calls.append((role, user_id))
        return SimpleNamespace(user_id="validated-seller-id")

    def fake_process_chat(**kwargs):
        process_calls.update(kwargs)
        return {
            "answer": AIMessage(content="Your sales look good."),
            "transcript": "",
            "tables": [],
            "activity": ["SQL specialist is analyzing Olist data"],
            "message_flow": [],
            "elapsed_seconds": 1.25,
            "token_usage": None,
        }

    monkeypatch.setattr(main, "make_scope", fake_scope)
    monkeypatch.setattr(main, "process_chat", fake_process_chat)
    response = TestClient(main.app).post(
        "/v1/chat",
        data={
            "message": "How are my sales?",
            "role": "seller",
            "user_id": "seller-from-login",
            "history": json.dumps([
                {"role": "user", "content": "Earlier question"},
                {"role": "assistant", "content": "Earlier answer"},
            ]),
        },
    )

    assert response.status_code == 200
    assert scope_calls == [("seller", "seller-from-login")]
    assert process_calls["user_id"] == "validated-seller-id"
    assert [message.content for message in process_calls["history"]] == [
        "Earlier question",
        "Earlier answer",
    ]
    assert response.json()["answer"] == "Your sales look good."
    assert response.json()["activity"] == ["SQL specialist is analyzing Olist data"]


def test_chat_rejects_unpaired_history():
    response = TestClient(main.app).post(
        "/v1/chat",
        data={
            "message": "Question",
            "role": "customer",
            "history": json.dumps([{"role": "user", "content": "Only one side"}]),
        },
    )

    assert response.status_code == 422


def test_chat_rejects_unknown_seller_before_running_agents(monkeypatch):
    def reject_seller(role, user_id=None):
        raise ValueError("Unknown seller id")

    def fail_if_agent_runs(**kwargs):
        raise AssertionError("agent must not run")

    monkeypatch.setattr(main, "make_scope", reject_seller)
    monkeypatch.setattr(main, "process_chat", fail_if_agent_runs)
    response = TestClient(main.app).post(
        "/v1/chat",
        data={"message": "Show my sales", "role": "seller", "user_id": "invalid"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Unknown seller id"


def test_chat_forwards_uploaded_image_to_agent_service(monkeypatch):
    calls = {}

    def fake_process_chat(**kwargs):
        calls.update(kwargs)
        return {
            "answer": AIMessage(content="This looks like a product."),
            "transcript": "",
            "tables": [],
            "activity": [],
            "message_flow": [],
            "elapsed_seconds": 0.1,
            "token_usage": None,
        }

    monkeypatch.setattr(main, "process_chat", fake_process_chat)
    response = TestClient(main.app).post(
        "/v1/chat",
        data={"message": "", "role": "customer"},
        files={"image": ("item.png", b"image-bytes", "image/png")},
    )

    assert response.status_code == 200
    assert calls["image"] == b"image-bytes"
    assert calls["image_suffix"] == ".png"
