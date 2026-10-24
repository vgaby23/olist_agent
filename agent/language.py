"""
language.py - guess which language a question is written in (English, Bahasa Indonesia, Portuguese)
Shared by the review (RAG) agent and the SQL agent, so both answer in the user's language.
"""
import re

# Words that tell the three languages of this project apart. Words that exist in two of them
# ("a", "no", "do", "me", "para", "mau") are left out on purpose.
LANGUAGE_WORDS = {
    "English": set("the what which who how are is there any anyone about show find give did customers customer "
                   "say said with from only most common overall average per your ignore instructions system "
                   "prompt and or for of in to my they their received ordered mention specific complaints "
                   "complain why when where does have has were was that".split()),
    "Bahasa Indonesia": set("apakah apa yang yg dan untuk dengan dari ada gak tidak nggak ga bgt banget terkait "
                            "produk rusak berikan tolong bagaimana kenapa mengapa pelanggan ulasan terburuk "
                            "terbaik datengnya datangnya telat atau saja semua kasih bintang pengiriman barang "
                            "kiriman bisa minta cari tampilkan ini itu di ke saya aku kamu".split()),
    "Portuguese": set("quais qual quem onde como porque não nao uma uns umas elogios reclamações reclamacoes "
                      "produtos qualidade clientes sobre dos das pelo pela com mais muito há existe algum "
                      "alguma tem entrega pedido são sao é foi".split()),
}

EXPLICIT_LANGUAGE = re.compile(
    r"\b(?:(?:answer|respond|reply)\s+in|answer\s+language\s*:)\s*"
    r"(english|bahasa\s+indonesia|indonesian|portuguese)\b",
    re.IGNORECASE,
)


def guess_language(text):
    """"What do customers say?" -> "English". Returns None when it is not clear.
    Counts how many words of the text belong to each language's list and takes the clear winner.
    (The LLM alone sometimes picked Portuguese for an English question, so we do not leave it to the LLM.)"""
    words = re.findall(r"[^\W\d_]+", text.lower())
    scores = {language: sum(1 for word in words if word in known) for language, known in LANGUAGE_WORDS.items()}
    best = max(scores.values())
    winners = [language for language, score in scores.items() if score == best]
    return winners[0] if best > 0 and len(winners) == 1 else None


def answer_language(text, default="English"):
    """Resolve a supported explicit language request, then detect text language, then use a default."""
    explicit = EXPLICIT_LANGUAGE.search(text or "")
    if explicit:
        language = explicit.group(1).lower()
        if language in {"bahasa indonesia", "indonesian"}:
            return "Bahasa Indonesia"
        return language.title()
    return guess_language(text or "") or default