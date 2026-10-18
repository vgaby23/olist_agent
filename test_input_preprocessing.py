"""Tests for input_preprocessing.py. OpenAI is never called: the two AI steps are replaced by fakes."""
import pytest

import input_preprocessing as pre


@pytest.fixture(autouse=True)
def categories(monkeypatch):
    monkeypatch.setattr(pre, "CATEGORIES", ["health_beauty", "toys"])


@pytest.fixture
def files(tmp_path):
    image, audio = tmp_path / "a.jpg", tmp_path / "a.m4a"
    image.write_bytes(b"x")
    audio.write_bytes(b"x")
    return str(image), str(audio)


def fake_guess(category="toys", is_product=True):
    return lambda path: pre.ProductGuess(is_product=is_product, category=category,
                                         description_en="red toy car", description_pt="carrinho vermelho")


def test_text_only_is_kept_as_is():
    assert pre.build_question(text="  hello  ") == "hello"


def test_nothing_sent_is_a_friendly_error():
    with pytest.raises(pre.InputError, match="Please send"):
        pre.build_question()


def test_voice_becomes_the_question(files, monkeypatch):
    monkeypatch.setattr(pre, "voice_to_text", lambda path, language=None: "apa keluhan soal mainan")
    assert pre.build_question(audio_path=files[1]) == "apa keluhan soal mainan"


def test_photo_only_asks_the_obvious_question(files, monkeypatch):
    monkeypatch.setattr(pre, "image_to_guess", fake_guess())
    q = pre.build_question(image_path=files[0], answer_language="Bahasa Indonesia")
    assert q.startswith("What do customers say about this product: red toy car (carrinho vermelho)?")
    assert "Product category: toys." in q and q.endswith("Answer in Bahasa Indonesia.")


def test_text_plus_photo_keeps_the_question_and_adds_the_photo(files, monkeypatch):
    monkeypatch.setattr(pre, "image_to_guess", fake_guess())
    q = pre.build_question(text="is it good?", image_path=files[0])
    assert q.startswith("is it good?\n(The photo shows: red toy car")
    assert "Answer in" not in q


def test_answer_language_not_added_twice(files, monkeypatch):
    monkeypatch.setattr(pre, "image_to_guess", fake_guess())
    q = pre.build_question(text="is it good? Answer in English.", image_path=files[0], answer_language="Bahasa Indonesia")
    assert "Bahasa" not in q


def test_invented_category_is_dropped(files, monkeypatch):
    monkeypatch.setattr(pre, "image_to_guess", fake_guess(category="flying_carpets"))
    assert "Product category" not in pre.build_question(image_path=files[0])


def test_photo_without_product_is_refused(files, monkeypatch):
    monkeypatch.setattr(pre, "image_to_guess", fake_guess(is_product=False))
    with pytest.raises(pre.InputError, match="can't see a product"):
        pre.build_question(image_path=files[0])


def test_file_checks(tmp_path):
    with pytest.raises(pre.InputError, match="can't find"):
        pre.check_file(tmp_path / "nope.jpg", pre.IMAGE_TYPES, "image")
    bad = tmp_path / "a.txt"
    bad.write_text("x")
    with pytest.raises(pre.InputError, match="not supported"):
        pre.check_file(bad, pre.IMAGE_TYPES, "image")


def test_voice_to_text_calls_whisper(files, monkeypatch):
    import openai
    seen = {}

    class FakeTranscriptions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return type("R", (), {"text": "  halo  "})()

    class FakeClient:
        audio = type("A", (), {"transcriptions": FakeTranscriptions()})()

    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: FakeClient())
    assert pre.voice_to_text(files[1], language="id") == "halo"
    assert seen["model"] == pre.STT_MODEL and seen["language"] == "id"


def test_silent_audio_is_an_error(files, monkeypatch):
    import openai

    class FakeTranscriptions:
        def create(self, **kwargs):
            return type("R", (), {"text": " "})()

    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: type("C", (), {
        "audio": type("A", (), {"transcriptions": FakeTranscriptions()})()})())
    with pytest.raises(pre.InputError, match="couldn't hear"):
        pre.voice_to_text(files[1])


def test_ask_with_media_sends_the_question_to_the_agent(files, monkeypatch):
    import rag_agent
    monkeypatch.setattr(pre, "image_to_guess", fake_guess())
    sent = []
    monkeypatch.setattr(rag_agent, "ask_rag_agent", lambda q, show_steps=False: sent.append(q) or "answer")
    assert pre.ask_with_media(image_path=files[0]) == "answer"
    assert "red toy car" in sent[0]


def test_ask_with_media_never_crashes(files, monkeypatch):
    assert "Please send" in pre.ask_with_media()
    monkeypatch.setattr(pre, "image_to_guess", lambda p: (_ for _ in ()).throw(ConnectionError("offline")))
    assert "ConnectionError" in pre.ask_with_media(image_path=files[0])


def test_command_line_reader():
    options, flags = pre.read_options(["--text", "hi", "--dry-run"])
    assert options["text"] == "hi" and flags["--dry-run"] is True
    with pytest.raises(SystemExit):
        pre.read_options(["--bogus"])