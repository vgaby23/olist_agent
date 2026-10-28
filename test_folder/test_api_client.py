import httpx

from utils import api_client


def test_validate_seller_does_not_send_authorization_header(monkeypatch):
    monkeypatch.setattr(api_client, "get_api_url", lambda: "https://api.example.com")
    request_details = {}

    def fake_post(url, **kwargs):
        request_details["url"] = url
        request_details.update(kwargs)
        return httpx.Response(200, json={"user_id": "seller-123"})

    monkeypatch.setattr(api_client.httpx, "post", fake_post)

    result = api_client.validate_seller_id("seller-123")

    assert result == "seller-123"
    assert request_details["url"] == "https://api.example.com/v1/seller/validate"
    assert request_details["data"] == {"seller_id": "seller-123"}
    assert "Authorization" not in request_details.get("headers", {})


def test_api_url_uses_environment_without_parsing_other_settings(monkeypatch):
    monkeypatch.setenv("API_URL", "http://localhost:8000/")
    monkeypatch.setenv("OPENAI_API_KEY", "not-an-api-url-setting")
    monkeypatch.setattr(
        api_client.st,
        "secrets",
        {},
    )

    assert api_client.get_api_url() == "http://localhost:8000"
