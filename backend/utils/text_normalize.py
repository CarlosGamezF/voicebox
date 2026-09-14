"""Word-level normalisation shared by transcript checks and the evaluation harness."""

import re
import unicodedata

# Spoken forms of abbreviations Whisper writes out in full.
SPANISH_ABBREVIATIONS = {
    "sr": "señor",
    "sra": "señora",
    "srta": "señorita",
    "dr": "doctor",
    "dra": "doctora",
    "num": "número",
    "nº": "número",
    "ud": "usted",
    "uds": "ustedes",
    "pag": "página",
    "avda": "avenida",
    "dpto": "departamento",
    "aprox": "aproximadamente",
    "etc": "etcétera",
}


def normalize_words(text: str) -> list[str]:
    """Lowercase words without punctuation or accents (ñ kept), abbreviations expanded."""
    text = text.lower().replace("ñ", "\x00")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.replace("\x00", "ñ")
    text = re.sub(r"[^a-z0-9ñº\s]", " ", text)
    words = []
    for word in text.split():
        expanded = SPANISH_ABBREVIATIONS.get(word, word)
        words.extend(normalize_words(expanded) if " " in expanded else [expanded])
    return words


def normalize_text(text: str) -> str:
    return " ".join(normalize_words(text))
