"""
input_preprocessing.py - turn a VOICE message or a PHOTO into plain text for the RAG agent
===========================================================================================

The agent only understands text. This file is the "front door" that converts everything else:

    voice file  --(1) speech-to-text (Whisper)-------------->  words
    photo       --(2) vision model: what product is this?--->  description + category
    typed text  ---------------------------------------------> kept as it is
                                       |
                                       v
                    (3) build_question(...) glues them into ONE question (plain text)
                                       |
                                       v
                    (4) ask_with_media(...) hands that question to rag_agent.ask_rag_agent

HOW TO RUN (needs OPENAI_API_KEY in .env)
    python input_preprocessing.py --text "what do people say about this?" --image shoe.jpg
    python input_preprocessing.py --audio question.m4a
    python input_preprocessing.py --image lamp.png --language "Bahasa Indonesia"
    add --dry-run to only see the question that WOULD be sent (no search, no agent answer)

WHY the category is only a hint, never a filter: a vision model can guess wrong. If we filtered on a
wrong category, the search would find nothing. So the guessed category goes into the question as a
hint ("product category: toys") and the agent decides whether to use it. A category that is not in
categories.json is thrown away, so a made-up name can never reach the search.
"""
import base64
import json
import logging
import mimetypes
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()      # reads OPENAI_API_KEY (and optional settings) from .env

from pydantic import BaseModel, Field                    # noqa: E402

# =============================================================================
# Part 1: SETTINGS
# =============================================================================
VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-4o-mini")   # any OpenAI chat model that can see images
STT_MODEL = os.environ.get("STT_MODEL", "whisper-1")           # speech-to-text model
# Spoken language as an ISO code. Without it Whisper guesses, and it often mis-heard Indonesian (even as Thai).
# Set STT_LANGUAGE=auto in .env to let Whisper detect the language again.
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "id")
# A few words in the expected style help Whisper spell the domain words right (it is a hint, not a filter).
STT_PROMPT = os.environ.get("STT_PROMPT", "Olist, ulasan, pembeli, penjual, sepatu, elektronik, kecantikan, "
                                          "pengiriman, reais, Sao Paulo, Rio de Janeiro.")
MAX_FILE_MB = 20                                               # refuse huge files before paying for them
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
AUDIO_TYPES = {".mp3", ".mp4", ".m4a", ".wav", ".webm", ".mpeg", ".mpga", ".ogg", ".flac"}


def find_categories():
    """Load the list of valid product categories (categories.json is written by ingest.py)."""
    here = Path(__file__).resolve().parent
    candidates = [os.environ.get("CATEGORIES_FILE"), "categories.json",
                  here / "categories.json", here.parent / "categories.json", here.parent.parent / "categories.json"]
    for path in candidates:
        if path and Path(path).is_file():
            return json.loads(Path(path).read_text(encoding="utf-8"))
    return []


CATEGORIES = find_categories()
log = logging.getLogger(__name__)


class InputError(Exception):
    """A problem with the file the user sent. The message is written for the user."""


def check_file(path, allowed_types, kind):
    """Stop early (with a friendly message) if the file is missing, the wrong type or too big."""
    file = Path(path)
    if not file.is_file():
        raise InputError(f"I can't find the {kind} file: {path}")
    if file.suffix.lower() not in allowed_types:
        raise InputError(f"The {kind} file type '{file.suffix}' is not supported. "
                         f"Use one of: {', '.join(sorted(allowed_types))}")
    if file.stat().st_size > MAX_FILE_MB * 1024 * 1024:
        raise InputError(f"The {kind} file is bigger than {MAX_FILE_MB} MB. Please send a smaller one.")
    return file


# =============================================================================
# Part 2: VOICE -> TEXT
# =============================================================================
def voice_to_text(audio_path, language=None):
    """Transcribe a voice file. `language` is an ISO code like "id" or "en" (helps accuracy);
    without it STT_LANGUAGE is used, and "auto" lets Whisper detect the language."""
    file = check_file(audio_path, AUDIO_TYPES, "audio")
    from openai import OpenAI

    options = {"model": STT_MODEL}
    language = (language or STT_LANGUAGE or "").strip().lower()
    if language and language != "auto":
        options["language"] = language
    if STT_PROMPT:
        options["prompt"] = STT_PROMPT
    with open(file, "rb") as audio:
        result = OpenAI().audio.transcriptions.create(file=audio, **options)
    text = result.text.strip()
    # Logged so a wrong answer can be traced back to a mis-heard question.
    log.info("Voice transcript (%s, language=%s): %r", STT_MODEL, options.get("language", "auto"), text)
    if not text:
        raise InputError("I couldn't hear any words in the audio. Please try again.")
    return text


# =============================================================================
# Part 3: PHOTO -> DESCRIPTION + CATEGORY
# =============================================================================
class ProductGuess(BaseModel):
    """What the vision model must fill in (structured output = no free-form text to parse)."""
    is_product: bool = Field(description="False if the photo does not show a product (a selfie, a landscape...)")
    category: str = Field(description="The closest category from the allowed list, copied exactly. "
                                      "Empty text if nothing fits.")
    description_en: str = Field(description="Short English description: product type, color, material, style")
    description_pt: str = Field(description="The same description in Brazilian Portuguese")


def image_to_guess(image_path):
    """Ask the vision model what product is in the photo. Returns a ProductGuess."""
    file = check_file(image_path, IMAGE_TYPES, "image")
    from langchain_core.messages import HumanMessage

    from agent.llm import chat_model

    mime = mimetypes.guess_type(str(file))[0] or "image/jpeg"
    picture = base64.b64encode(file.read_bytes()).decode()
    instruction = "Identify the product in this photo."
    if CATEGORIES:
        instruction += " Choose the closest category from this list (copy it exactly): " + ", ".join(CATEGORIES)
    message = HumanMessage(content=[
        {"type": "text", "text": instruction},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{picture}"}},
    ])
    model = chat_model(VISION_MODEL, temperature=0).with_structured_output(ProductGuess)
    return model.invoke([message])


def clean_category(category):
    """Keep the category only if it is really in our list; otherwise None (never trust a made-up one)."""
    if category in CATEGORIES:
        return category
    return None


# =============================================================================
# Part 4: GLUE EVERYTHING INTO ONE QUESTION
# =============================================================================
def build_question(text=None, audio_path=None, image_path=None, answer_language=None, audio_language=None):
    """Return ONE plain-text question for the agent, from any mix of typed text, voice and photo.

    answer_language: e.g. "Bahasa Indonesia". Useful when only a photo is sent, because then nothing
                     tells the agent which language the user speaks (without it, English is used).
    """
    typed = (text or "").strip()
    spoken = voice_to_text(audio_path, audio_language) if audio_path else ""
    words = " ".join(part for part in (typed, spoken) if part)      # what the user SAID or typed

    if not image_path:
        if not words:
            raise InputError("Please send a question as text, a voice message or a photo.")
        return words

    guess = image_to_guess(image_path)
    if not guess.is_product:
        raise InputError("I can't see a product in this photo. Please send a clearer photo of the item.")

    category = clean_category(guess.category)
    about = f"{guess.description_en} ({guess.description_pt})"      # reviews are in Portuguese
    hint = f" Product category: {category}." if category else ""

    if words:    # the user asked something AND showed a photo ("is this good?")
        question = f"{words}\n(The photo shows: {about}.{hint})"
    else:        # photo only: ask the obvious question
        question = f"What do customers say about this product: {about}?{hint}"

    if answer_language and "answer in" not in question.lower():
        question += f" Answer in {answer_language}."
    return question


def ask_with_media(text=None, audio_path=None, image_path=None, answer_language=None,
                   audio_language=None, show_steps=False):
    """The whole flow: voice / photo / text -> one question -> the RAG agent -> answer text."""
    try:
        question = build_question(text, audio_path, image_path, answer_language, audio_language)
    except InputError as problem:
        return str(problem)
    except Exception as error:               # e.g. no internet, bad key: say it in words, don't crash
        return f"I couldn't process your voice/photo: {type(error).__name__}: {str(error)[:200]}"

    from agent import rag_agent                          # imported here so this file also works without Qdrant set up
    return rag_agent.ask_rag_agent(question, show_steps=show_steps)


# =============================================================================
# Part 5: TALKING TO IT FROM THE TERMINAL
# =============================================================================
def read_options(args):
    """Tiny argument reader: --text X --audio F --image F --language L --audio-language C [--dry-run] [--show-steps]"""
    options = {"text": None, "audio": None, "image": None, "language": None, "audio-language": None}
    flags = {"--dry-run": False, "--show-steps": False}
    index = 0
    while index < len(args):
        name = args[index]
        if name in flags:
            flags[name] = True
            index += 1
        elif name.startswith("--") and name[2:] in options and index + 1 < len(args):
            options[name[2:]] = args[index + 1]
            index += 2
        else:
            raise SystemExit(f"Unknown or incomplete option: {name}\n" + __doc__)
    return options, flags


def main(argv=None):
    options, flags = read_options(list(sys.argv[1:] if argv is None else argv))
    if not any((options["text"], options["audio"], options["image"])):
        raise SystemExit(__doc__)

    if flags["--dry-run"]:
        try:
            print(build_question(options["text"], options["audio"], options["image"],
                                 options["language"], options["audio-language"]))
        except InputError as problem:
            print(problem)
        return

    print(ask_with_media(options["text"], options["audio"], options["image"], options["language"],
                         options["audio-language"], show_steps=flags["--show-steps"]))


if __name__ == "__main__":
    main()