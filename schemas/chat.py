from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Role = Literal["customer", "seller", "company"]


class ChatHistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(max_length=4000)
    role: Role
    user_id: str | None = Field(default=None, max_length=64)
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    transcript: str
    tables: list[dict]
    activity: list[str]
    message_flow: list[dict]
    elapsed_seconds: float
    token_usage: dict | None = None
    image_context: str | None = None

class SellerValidateResponse(BaseModel):
    success: bool = True
    message: str
    user_id: str