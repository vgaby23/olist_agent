import json
import sqlite3
from pathlib import Path
from typing import Optional, Set, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError
from fastapi.concurrency import run_in_threadpool

from dataset.preprocessing.numeric_data.sql_tools import make_scope
from schemas.chat import ChatRequest, ChatResponse, Role, SellerValidateResponse
from services.chat_service import process_chat
from utils import input_preprocessing
from utils.chat_text import clean_answer


app = FastAPI(title="Olist Virtual Assistant API", version="1.0.0")

MAX_FILE_SIZE_BYTES = input_preprocessing.MAX_FILE_MB * 1024 * 1024


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/v1/seller/validate", response_model=SellerValidateResponse)
async def validate_seller(seller_id: str = Form(...)):
    try:
        scope = make_scope("seller", seller_id)
    except ValueError as error:
        # Bad seller_id / invalid seller
        raise HTTPException(status_code=403, detail=str(error)) from error
    except (OSError, sqlite3.Error) as error:
        # DB issues
        raise HTTPException(status_code=503, detail="The Olist database is unavailable.") from error

    return SellerValidateResponse(
        message="Seller ID validated successfully.",
        user_id=scope.user_id,
    )



@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(default=""),
    role: Role = Form(...),
    user_id: str = Form(default=""),
    history: str = Form(default="[]"),
    image: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
):
    # Parse & validate body into a schema
    try:
        request = ChatRequest(
            message=message,
            role=role,
            user_id=user_id or None,
            history=json.loads(history),
        )
    except (json.JSONDecodeError, ValidationError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    # Business rules around role/user_id/history
    if request.role == "seller" and not request.user_id:
        raise HTTPException(status_code=422, detail="Seller requests require a user_id.")
    if request.role != "seller" and request.user_id is not None:
        raise HTTPException(status_code=422, detail="user_id is only valid for seller requests.")
    if len(request.history) % 2:
        raise HTTPException(
            status_code=422,
            detail="History must contain user/assistant message pairs.",
        )
    for index, item in enumerate(request.history):
        if not item.content.strip():
            raise HTTPException(status_code=422, detail="History messages cannot be blank.")
        expected_role = "user" if index % 2 == 0 else "assistant"
        if item.role != expected_role:
            raise HTTPException(
                status_code=422,
                detail="History must alternate user and assistant messages.",
            )

    # File handling (async)
    image_bytes, image_suffix = await read_upload(
        upload=image,
        kind="image",
        allowed_types=input_preprocessing.IMAGE_TYPES,
    )
    audio_bytes, audio_suffix = await read_upload(
        upload=audio,
        kind="audio",
        allowed_types=input_preprocessing.AUDIO_TYPES,
    )

    if not message.strip() and image_bytes is None and audio_bytes is None:
        raise HTTPException(status_code=422, detail="Provide a message, image, or audio file.")

    # Scope loading (DB access)
    try:
        scope = make_scope(request.role, request.user_id)
    except ValueError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(status_code=503, detail="The Olist database is unavailable.") from error

    # Convert history into LangChain messages
    messages = [
        HumanMessage(content=item.content)
        if item.role == "user"
        else AIMessage(content=item.content)
        for item in request.history
    ]

    try:
        result = await run_in_threadpool(
            process_chat,
            role=request.role,
            user_id=scope.user_id,
            message=request.message,
            history=messages,
            image=image_bytes,
            image_suffix=image_suffix,
            audio=audio_bytes,
            audio_suffix=audio_suffix,
        )
    except input_preprocessing.InputError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    answer = result["answer"]
    if answer is None:
        raise HTTPException(status_code=502, detail="The assistant did not return an answer.")

    # Serialize tables into JSON-friendly format
    tables = [
        {
            "columns": [str(column) for column in frame.columns],
            "rows": json.loads(frame.to_json(orient="records", date_format="iso")),
        }
        for frame in result["tables"]
    ]

    return ChatResponse(
        answer=clean_answer(answer.text),
        refused=bool(answer.additional_kwargs.get("violated_rule")),
        transcript=result["transcript"],
        tables=tables,
        activity=result["activity"],
        message_flow=result["message_flow"],
        elapsed_seconds=result["elapsed_seconds"],
        token_usage=result["token_usage"],
    )


async def read_upload(
    upload: Optional[UploadFile],
    kind: str,
    allowed_types: Set[str],
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Read an uploaded file (if any), validate its extension and size.
    Returns (content_bytes, suffix) or (None, None) if no file.
    """
    if upload is None:
        return None, None

    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported {kind} file type. Use one of: {', '.join(sorted(allowed_types))}",
        )

    content = await upload.read(MAX_FILE_SIZE_BYTES + 1)
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"The {kind} file is bigger than {input_preprocessing.MAX_FILE_MB} MB.",
        )

    return content, suffix
