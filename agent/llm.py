"""
llm.py - ONE place to create the OpenAI chat model for every agent

Reasoning models (o1/o3/o4, gpt-5, gpt-6 such as gpt-6-luna, ...) reject `temperature`: the API answers
"Unsupported parameter: 'temperature' is not supported with this model". For those models we simply do not
send it. Older chat models (gpt-4o-mini, gpt-4.1, ...) still get temperature=0 (least random answers).

If a new model is wrongly detected, override it in .env:
    LLM_SUPPORTS_TEMPERATURE=false     never send temperature
    LLM_SUPPORTS_TEMPERATURE=true      always send it
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-6-luna"

# Model names starting with these do not accept temperature (only the default sampling).
NO_TEMPERATURE_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-6")
# ... except these chat variants, which do accept it.
TEMPERATURE_OK_PREFIXES = ("gpt-5-chat",)


def supports_temperature(model):
    """True when the model accepts a temperature setting. LLM_SUPPORTS_TEMPERATURE in .env overrides the guess."""
    override = os.getenv("LLM_SUPPORTS_TEMPERATURE", "").strip().lower()
    if override in ("true", "1", "yes"):
        return True
    if override in ("false", "0", "no"):
        return False
    name = (model or "").strip().lower()
    if name.startswith(TEMPERATURE_OK_PREFIXES):
        return True
    return not name.startswith(NO_TEMPERATURE_PREFIXES)


def chat_model(model=None, temperature=0, **kwargs):
    """ChatOpenAI for `model` (default: LLM_MODEL from .env, else gpt-6-luna).
    `temperature` is only sent when the model supports it."""
    model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
    if temperature is not None and supports_temperature(model):
        kwargs["temperature"] = temperature
    return ChatOpenAI(model=model, **kwargs)
