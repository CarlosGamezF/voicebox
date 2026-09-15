"""Spell out what the TTS model misreads before synthesis.

Measured with Whisper on Spanish clones: digits, dates, money and
abbreviations were read wrong ("1.234,56 €" became "un millón treinta y
cuatro mil quinientos euros", "15/09/2026" became "15 de noviembre",
"p. ej." became "pa es"). Plain prose had no errors, so the fix is to hand
the model the words a speaker would say and leave everything else exactly
as written: punctuation, casing, line breaks, names, English text.

Only Spanish (Spain conventions) is implemented; other languages pass
through unchanged.
"""

import re
from collections.abc import Callable

from .spanish_numbers import cardinal, decimal_words, digits, ordinal

MONTHS_ES = [
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
]

# Abbreviations with a trailing period, keyed without it. Capitalisation of
# the expansion follows the sentence (see _capitalize_like_sentence_start).
ABBREVIATIONS_ES = {
    "sr": "señor",
    "sra": "señora",
    "srta": "señorita",
    "sres": "señores",
    "dr": "doctor",
    "dra": "doctora",
    "dña": "doña",
    "ud": "usted",
    "uds": "ustedes",
    "vd": "usted",
    "vds": "ustedes",
    "etc": "etcétera",
    "aprox": "aproximadamente",
    "dpto": "departamento",
    "avda": "avenida",
    "tel": "teléfono",
    "pág": "página",
    "págs": "páginas",
    "art": "artículo",
    "cap": "capítulo",
    "fig": "figura",
    "vol": "volumen",
    "núm": "número",
}
# Titles precede a name, so their period never closes a sentence.
_TITLES_ES = {"sr", "sra", "srta", "sres", "dr", "dra", "dña"}
# Abbreviations that keep their capital letters when expanded.
PROPER_ABBREVIATIONS_ES = {"ee. uu": "Estados Unidos", "ee.uu": "Estados Unidos"}

UNITS_ES = {
    "km/h": ("kilómetro por hora", "kilómetros por hora"),
    "km": ("kilómetro", "kilómetros"),
    "cm": ("centímetro", "centímetros"),
    "mm": ("milímetro", "milímetros"),
    "m": ("metro", "metros"),
    "kg": ("kilo", "kilos"),
    "mg": ("miligramo", "miligramos"),
    "g": ("gramo", "gramos"),
    "ml": ("mililitro", "mililitros"),
    "l": ("litro", "litros"),
    "h": ("hora", "horas"),
    "min": ("minuto", "minutos"),
    "s": ("segundo", "segundos"),
    "°c": ("grado", "grados"),
    "ºc": ("grado", "grados"),
}
CURRENCIES_ES = {
    "€": ("euro", "euros", "céntimo", "céntimos"),
    "$": ("dólar", "dólares", "centavo", "centavos"),
    "£": ("libra", "libras", "penique", "peniques"),
}

_NUM = r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+\.\d{1,2}(?!\d)|\d+\.\d{4,}|\d+(?:,\d+)?"
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(UNITS_ES, key=len, reverse=True))
_ABBR_ALT = "|".join(re.escape(a) for a in sorted(ABBREVIATIONS_ES, key=len, reverse=True))

_RULES_ES: list[tuple[re.Pattern, Callable[[re.Match], str]]] = []


def _rule(pattern: str):
    def register(fn):
        _RULES_ES.append((re.compile(pattern, re.IGNORECASE | re.UNICODE), fn))
        return fn

    return register


def verbalize(text: str, language: str | None) -> str:
    """Return *text* as it should be spoken in *language*; unchanged for languages without rules."""
    if language and language.lower().startswith("es"):
        return verbalize_es(text)
    return text


def verbalize_es(text: str) -> str:
    """Spanish verbalization: one left-to-right scan, first matching rule wins at each position."""
    master = re.compile(
        "|".join(f"(?P<r{i}>{p.pattern})" for i, (p, _) in enumerate(_RULES_ES)), re.IGNORECASE | re.UNICODE
    )
    out: list[str] = []
    pos = 0
    for m in master.finditer(text):
        index = next(i for i, (_p, _) in enumerate(_RULES_ES) if m.group(f"r{i}") is not None)
        pattern, handler = _RULES_ES[index]
        inner = pattern.match(text, m.start())
        replacement = handler(inner)
        if replacement is None:
            continue
        out.append(text[pos : m.start()])
        out.append(_capitalize_like_sentence_start("".join(out[-2:]), replacement))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def _capitalize_like_sentence_start(before: str, replacement: str) -> str:
    """A sentence-initial number keeps the sentence capitalised: "15 personas" -> "Quince personas"."""
    before = before.rstrip(" \t\"'«“‘(¿¡")  # noqa: RUF001 -- typographic quotes are intended
    if not before or before[-1] in ".!?\n…":
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _number_words(token: str, feminine: bool = False) -> str:
    """Words for one numeric token in Spanish notation: "1.234,56", "12,5", "1.5", "300"."""
    token = token.replace(" ", "")
    if "," in token:
        whole, fraction = token.split(",", 1)
        return decimal_words(whole.replace(".", ""), fraction, feminine)
    if "." in token:
        parts = token.split(".")
        if all(len(p) == 3 for p in parts[1:]):  # thousands separators
            return cardinal(int("".join(parts)), feminine)
        return decimal_words(parts[0], "".join(parts[1:]), feminine)
    if len(token) >= 10:  # phone numbers and codes: digit by digit
        return digits(token)
    return cardinal(int(token), feminine)


def _amount(token: str) -> tuple[int, str | None]:
    """Split a money token into whole units and the cents string, if any."""
    token = token.replace(".", "")
    if "," in token:
        whole, cents = token.split(",", 1)
        return int(whole or 0), cents
    return int(token), None


# --- rules, in priority order --------------------------------------------------------------------


@_rule(r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.]+|#\w+")
def _protect_links(m: re.Match) -> str:
    return m.group(0)


@_rule(
    r"\bv\d+(?:\.\d+)+\b|\b(?!\d{1,3}(?:\.\d{3})+\b)\d+\.\d+\.\d+\b"
    r"|\b[A-Za-z]+\d[\w-]*\b|\b[A-Za-z]{2,}-\d+\b|\b\d+/\d+\b(?![/-]\d)"
)
def _protect_identifiers(m: re.Match) -> str:
    return m.group(0)  # version numbers, model names (Qwen3, GPT-4o, COVID-19), fractions and 24/7


@_rule(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
def _date_dmy(m: re.Match) -> str:
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return _date_words(day, month, year) or m.group(0)


@_rule(r"\b(\d{4})-(\d{2})-(\d{2})\b")
def _date_iso(m: re.Match) -> str:
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return _date_words(day, month, year) or m.group(0)


def _date_words(day: int, month: int, year: int) -> str | None:
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    return f"{cardinal(day)} de {MONTHS_ES[month - 1]} de {cardinal(year)}"


@_rule(r"\b(\d{1,2}):(\d{2})\b(?:\s?h\b)?")
def _time(m: re.Match) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23 or minute > 59:
        return m.group(0)
    minutes = "" if minute == 0 else (" cero " + cardinal(minute) if minute < 10 else " " + cardinal(minute))
    return f"{cardinal(hour)}{minutes}".strip()


_ROMAN_NOUNS = {"vol.": "volumen", "cap.": "capítulo"}


@_rule(r"\b(siglo|capítulo|tomo|parte|volumen|vol\.|cap\.)\s+([IVXLC]+)\b")
def _roman_after_noun(m: re.Match) -> str:
    value = _roman_to_int(m.group(2).upper())
    noun = _ROMAN_NOUNS.get(m.group(1).lower(), m.group(1))
    return f"{noun} {cardinal(value)}" if value else m.group(0)


@_rule(r"\b(\d{1,3})\.?(º|ª|°|er)\b")
def _ordinal(m: re.Match) -> str:
    n = int(m.group(1))
    marker = m.group(2).lower()
    return ordinal(n, feminine=marker == "ª", apocopate=marker == "er")


@_rule(rf"(?:({_NUM})\s?(€|\$|£|euros?\b)|(€|\$|£)\s?({_NUM}))")
def _money(m: re.Match) -> str:
    token, symbol = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
    names = CURRENCIES_ES.get(symbol[:1] if symbol[:1] in CURRENCIES_ES else "€", CURRENCIES_ES["€"])
    whole, cents = _amount(token)
    amount = "un" if whole == 1 else cardinal(whole)
    if _round_millions(whole):
        amount += " de"
    words = f"{amount} {names[0] if whole == 1 else names[1]}"
    if symbol.startswith("£") and whole == 1:
        words = f"una {names[0]}"
    if cents and int(cents):
        cents_value = int(cents.ljust(2, "0")[:2])
        words += (
            f" con {'un' if cents_value == 1 else cardinal(cents_value)} {names[2] if cents_value == 1 else names[3]}"
        )
    return words


@_rule(rf"(?<![\w.,])(-?)({_NUM})\s?%")
def _percent(m: re.Match) -> str:
    return f"{_signed(m.group(1), _number_words(m.group(2)))} por ciento"


@_rule(rf"(?<![\w.,])(-?)({_NUM})\s?({_UNIT_ALT})(?![\wáéíóú])")
def _unit(m: re.Match) -> str:
    return _signed(m.group(1), _quantity(m.group(2), m.group(3)))


@_rule(r"\b\d{2,4}(?:-\d{2,4}){2,}\b")
def _hyphenated_groups(m: re.Match) -> str:
    return ", ".join(cardinal(int(g)) for g in m.group(0).split("-"))  # phone numbers written 912-345-678


@_rule(rf"(?<![\d.,-])(\d{{1,4}})-(\d{{1,4}})(?![\d-]|[.,]\d)(?:\s?({_UNIT_ALT})(?![\wáéíóú]))?")
def _range(m: re.Match) -> str:
    low, high, unit = m.group(1), m.group(2), m.group(3)
    if unit:
        return f"{cardinal(int(low))} a {_quantity(high, unit)}"
    return f"{cardinal(int(low))} a {cardinal(int(high))}"


def _quantity(token: str, unit: str) -> str:
    singular, plural = UNITS_ES[unit.lower()]
    one = token in ("1", "1,0", "1,00")
    return f"{'un' if one else _number_words(token)} {singular if one else plural}"


def _signed(sign: str, words: str) -> str:
    return f"menos {words}" if sign else words


@_rule(rf"(?<![\w.,])(-?)({_NUM})(?![\w])")
def _number(m: re.Match) -> str:
    words = _number_words(m.group(2))
    if _round_millions_token(m.group(2)) and _followed_by_noun(m):
        words += " de"  # "dos millones de habitantes"
    return _signed(m.group(1), words)


def _round_millions(n: int) -> bool:
    return n >= 10**6 and n % 10**6 == 0


def _round_millions_token(token: str) -> bool:
    plain = token.replace(".", "")
    return plain.isdigit() and _round_millions(int(plain))


_NOT_NOUNS = {"de", "y", "o", "u", "e", "con", "sin", "en", "a", "al", "del", "por", "para", "más", "menos"}


def _followed_by_noun(m: re.Match) -> bool:
    rest = m.string[m.end() :]
    word = re.match(r"\s+([a-záéíóúñ]+)", rest)
    return bool(word) and word.group(1) not in _NOT_NOUNS


@_rule(r"\bEE\.\s?UU\.")
def _estados_unidos(m: re.Match) -> str:
    return "Estados Unidos" + _sentence_period(m)


@_rule(r"\bp\.\s?ej\.")
def _por_ejemplo(m: re.Match) -> str:
    return "por ejemplo" + _sentence_period(m)


@_rule(r"\bD\.(?=\s+[A-ZÁÉÍÓÚ])")
def _don(m: re.Match) -> str:
    return "don"


@_rule(r"\bc/(?=\s?[A-ZÁÉÍÓÚ])")
def _calle(m: re.Match) -> str:
    return "calle"


@_rule(r"\bs/n\b")
def _sin_numero(m: re.Match) -> str:
    return "sin número"


@_rule(rf"\b({_ABBR_ALT})\.(?!\w)")
def _abbreviation(m: re.Match) -> str:
    key = m.group(1).lower()
    period = "" if key in _TITLES_ES else _sentence_period(m)
    return ABBREVIATIONS_ES[key] + period


@_rule(r"\bn\.?º(?=\s?\d)")
def _numero_sign(m: re.Match) -> str:
    return "número"


def _sentence_period(m: re.Match) -> str:
    """Keep the period when the abbreviation also ended its sentence."""
    rest = m.string[m.end() :].lstrip(" \t\"'»”’)")  # noqa: RUF001 -- typographic quotes are intended
    if not rest or rest[0] in "\n" or rest[0].isupper() or rest[0] in "¿¡":
        return "."
    return ""


def _roman_to_int(s: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    for i, ch in enumerate(s):
        if ch not in values:
            return 0
        if i + 1 < len(s) and values[ch] < values[s[i + 1]]:
            total -= values[ch]
        else:
            total += values[ch]
    return total
