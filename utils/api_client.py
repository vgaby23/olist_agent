import json
import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError


load_dotenv()


class APIRequestError(Exception):
    """Error that can be shown to the user when an API request fails."""


def get_api_url() -> str:
    try:
        api_url = st.secrets.get("API_URL")
    except StreamlitSecretNotFoundError:
        api_url = None

    api_url = api_url or os.getenv("API_URL")
    if not api_url:
        raise APIRequestError("Set API_URL in Streamlit secrets or your local .env file.")

    return api_url.rstrip("/")


def post_to_api(path, data, files=None, timeout=120):
    try:
        response = httpx.post(
            get_api_url() + path,
            data=data,
            files=files,
            timeout=timeout,
        )
    except httpx.TimeoutException as error:
        raise APIRequestError("The assistant API timed out. Please try again.") from error
    except httpx.RequestError as error:
        raise APIRequestError(
            "Could not connect to the assistant API. Check API_URL and try again."
        ) from error

    if not response.is_success:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise APIRequestError(f"Assistant API returned {response.status_code}: {detail}")

    try:
        return response.json()
    except ValueError as error:
        raise APIRequestError("The assistant API returned an invalid response.") from error


def send_chat(
    *,
    message: str,
    role: str,
    user_id: str | None,
    history: list[dict],
    image: tuple[str, bytes] | None = None,
    audio: tuple[str, bytes] | None = None,
) -> dict:
    data = {
        "message": message,
        "role": role,
        "user_id": user_id or "",
        "history": json.dumps(history),
    }

    files = {}
    if image:
        name, content = image
        files["image"] = (Path(name).name, content)
    if audio:
        name, content = audio
        files["audio"] = (Path(name).name, content)

    return post_to_api("/v1/chat", data, files or None)


def validate_seller_id(user_id: str) -> str:
    result = post_to_api(
        "/v1/seller/validate",
        {"seller_id": user_id},
        timeout=15,
    )
    try:
        return result["user_id"]
    except (KeyError, TypeError) as error:
        raise APIRequestError("The assistant API returned an invalid seller ID response.") from error
