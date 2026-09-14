"""Word-level normalisation shared by transcript checks and the evaluation harness."""

import re
import unicodedata

# Spoken forms of abbreviations Whisper writes out in full, keyed by their
# punctuation-free lowercase spelling.
SPANISH_ABBREVIATIONS = {
    "sr": "señor",
    "sra": "señora",
    "srta": "señorita",
    "dr": "doctor",
    "dra": "doctora",
    "num": "número",
    "nº": "número",
    "n º": "número",
    "ud": "usted",
    "uds": "ustedes",
    "pag": "página",
    "pags": "páginas",
    "avda": "avenida",
    "dpto": "departamento",
    "aprox": "aproximadamente",
    "etc": "etcétera",
    "p ej": "por ejemplo",
    "ee uu": "estados unidos",
}


def _plain(text: str) -> str:
    """Lowercase, accent-free (ñ kept) text with punctuation turned into spaces."""
    text = text.lower().replace("ñ", "\x00")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.replace("\x00", "ñ")
    return " ".join(re.sub(r"[^a-z0-9ñº\s]", " ", text).split())


_MULTI_WORD = {k: _plain(v) for k, v in SPANISH_ABBREVIATIONS.items() if " " in k}
_MULTI_WORD_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _MULTI_WORD) + r")\b")


def normalize_words(text: str) -> list[str]:
    """Lowercase words without punctuation or accents (ñ kept), abbreviations expanded."""
    plain = _MULTI_WORD_RE.sub(lambda m: _MULTI_WORD[m.group(1)], _plain(text))
    words: list[str] = []
    for word in plain.split():
        expanded = SPANISH_ABBREVIATIONS.get(word)
        words.extend(_plain(expanded).split() if expanded else [word])
    return words


def normalize_text(text: str) -> str:
    """``normalize_words`` joined back into one space-separated string."""
    return " ".join(normalize_words(text))
