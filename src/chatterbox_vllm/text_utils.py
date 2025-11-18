import re
import unicodedata
from typing import Iterable, Optional


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _build_stopword_pattern(words: Iterable[str]) -> re.Pattern[str]:
    normalized_words = sorted({
        re.escape(_strip_accents(word.lower()))
        for word in words
        if word
    }, key=len, reverse=True)
    if not normalized_words:
        return re.compile(r"(?!x)")
    word_group = "|".join(normalized_words)
    return re.compile(rf"(?<!\w)(?:{word_group})(?!\w)")


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    if text[0].islower():
        text = text[0].upper() + text[1:]

    text = " ".join(text.split())

    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}

_SCRIPT_RANGES = {
    "zh": ((0x4E00, 0x9FFF), (0x3400, 0x4DBF)),
    "ja": ((0x3040, 0x309F), (0x30A0, 0x30FF)),
    "ko": ((0xAC00, 0xD7AF),),
    "ru": ((0x0400, 0x04FF),),
    "el": ((0x0370, 0x03FF),),
    "ar": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "he": ((0x0590, 0x05FF),),
    "hi": ((0x0900, 0x097F),),
}

_SPECIAL_CHAR_SETS = {
    "es": "áéíóúüñ¡¿",
    "fr": "àâæçéèêëîïôœùûüÿ",
    "de": "äöüß",
    "it": "àèéìíîòóù",
    "pt": "áâãàéêíóôõúç",
    "pl": "ąćęłńóśźż",
    "sv": "åäö",
    "fi": "åäö",
    "da": "æøå",
    "no": "æøå",
    "tr": "çğıöşü",
}

_STOPWORD_MAP = {
    "es": ("que", "de", "para", "como", "pero", "hola", "gracias", "cuando", "donde", "buenos", "tienes", "usted", "bienvenido", "buenas"),
    "fr": ("bonjour", "merci", "avec", "pour", "vous", "etre", "etes", "pas", "nous", "toujours", "aller"),
    "de": ("und", "nicht", "der", "die", "das", "aber", "danke", "bitte", "hallo", "mich", "doch"),
    "it": ("ciao", "grazie", "perche", "allora", "questo", "quello", "anche", "sempre", "sono", "bene"),
    "pt": ("ola", "obrigado", "voce", "nao", "porque", "tambem", "tudo", "bem", "vamos"),
    "nl": ("hallo", "dank", "als", "jij", "niet", "goed", "alstublieft", "gezellig"),
    "da": ("hej", "tak", "ikke", "bare", "mange", "venligst", "hvad"),
    "no": ("hei", "takk", "ikke", "bare", "mange", "venn", "hva", "hvordan"),
    "sv": ("hej", "tack", "inte", "bara", "manga", "snalla", "varsagod"),
    "fi": ("hei", "kiitos", "en", "olen", "paljon", "miksi", "kuinka", "ystava"),
    "pl": ("czesc", "dziekuje", "prosze", "jest", "nie", "dzien", "dobry", "jak", "dobrze"),
    "tr": ("merhaba", "tesekkur", "gorusuruz", "degil", "icin", "bunu", "sen", "ben", "kadar"),
    "ms": ("apa", "khabar", "selamat", "sudah", "tidak", "terima kasih", "saya", "anda", "kamu"),
    "sw": ("habari", "asante", "karibu", "rafiki", "safari", "wewe", "mimi", "sisi"),
}

_STOPWORD_PATTERNS = {
    lang: _build_stopword_pattern(words)
    for lang, words in _STOPWORD_MAP.items()
}


def _detect_by_script(text: str, supported: set[str]) -> Optional[str]:
    hits: dict[str, int] = {}
    for char in text:
        code = ord(char)
        for lang, ranges in _SCRIPT_RANGES.items():
            if lang not in supported:
                continue
            if any(start <= code <= end for start, end in ranges):
                hits[lang] = hits.get(lang, 0) + 1
                break
    if not hits:
        return None
    return max(hits, key=hits.get)


def _detect_by_special_chars(text: str, supported: set[str]) -> Optional[str]:
    lower_text = text.lower()
    best_lang: Optional[str] = None
    best_score = 0
    for lang, charset in _SPECIAL_CHAR_SETS.items():
        if lang not in supported:
            continue
        score = sum(lower_text.count(ch) for ch in charset)
        if score > best_score:
            best_lang = lang
            best_score = score
    if best_score == 0:
        return None
    return best_lang


def _detect_by_stopwords(text: str, supported: set[str]) -> Optional[str]:
    normalized = _strip_accents(text).lower()
    best_lang: Optional[str] = None
    best_score = 0
    match_cache: dict[str, list[str]] = {}
    for lang, pattern in _STOPWORD_PATTERNS.items():
        if lang not in supported:
            continue
        matches = pattern.findall(normalized)
        match_cache[lang] = matches
        score = len(matches)
        if score > best_score:
            best_score = score
            best_lang = lang
    if best_lang is None or best_score == 0:
        return None
    if best_score >= 2:
        return best_lang
    longest_match = max((len(match.strip()) for match in match_cache[best_lang]), default=0)
    if longest_match >= 5 or len(normalized) >= 30:
        return best_lang
    return None


def detect_language_from_text(
    text: str,
    supported_languages: Optional[Iterable[str]] = None,
    default: str = "en",
) -> str:
    supported = set(supported_languages or SUPPORTED_LANGUAGES.keys())
    if not supported:
        supported = {default}
    stripped = text.strip()
    if not stripped:
        return default if default in supported else next(iter(supported))
    script_lang = _detect_by_script(stripped, supported)
    if script_lang:
        return script_lang
    char_lang = _detect_by_special_chars(stripped, supported)
    if char_lang:
        return char_lang
    stopword_lang = _detect_by_stopwords(stripped, supported)
    if stopword_lang:
        return stopword_lang
    return default if default in supported else next(iter(supported))
