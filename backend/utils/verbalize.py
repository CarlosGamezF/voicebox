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
_CURRENCY_WORDS = {"euro": "€", "euros": "€", "dólar": "$", "dólares": "$", "libra": "£", "libras": "£"}
_FEMININE_NOUNS = {"hora", "libra"}
# Words after a number that are not the noun it counts ("un millón y medio").
_NOT_NOUNS = {"de", "y", "o", "u", "e", "con", "sin", "en", "a", "al", "del", "por", "para", "más", "menos"}

# Thousands groups "1.234", dot decimals "1.5"/"3.14159" (three digits after a dot are thousands),
# comma decimals "12,5" and plain digits.
_NUM = r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+\.\d{1,2}(?!\d)|\d+\.\d{4,}|\d+(?:,\d+)?"
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(UNITS_ES, key=len, reverse=True))
_ABBR_ALT = "|".join(re.escape(a) for a in sorted(ABBREVIATIONS_ES, key=len, reverse=True))
_NOT_IN_WORD = r"(?![\wáéíóú])"

_RULES_ES: list[tuple[re.Pattern, Callable[[re.Match], str]]] = []
_MASTER_ES: re.Pattern | None = None


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
    global _MASTER_ES
    if _MASTER_ES is None:
        _MASTER_ES = re.compile(
            "|".join(f"(?P<r{i}>{p.pattern})" for i, (p, _) in enumerate(_RULES_ES)), re.IGNORECASE | re.UNICODE
        )
    out: list[str] = []
    pos = 0
    for m in _MASTER_ES.finditer(text):
        index = next(i for i in range(len(_RULES_ES)) if m.group(f"r{i}") is not None)
        pattern, handler = _RULES_ES[index]
        inner = pattern.match(text, m.start())
        replacement = handler(inner)
        out.append(text[pos : m.start()])
        if replacement == inner.group(0):
            out.append(replacement)  # protected token: not even its casing changes
        else:
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


def _is_thousands(token: str) -> bool:
    parts = token.split(".")
    return len(parts) > 1 and all(len(p) == 3 for p in parts[1:])


def _number_words(token: str, feminine: bool = False, apocopate: bool = False) -> str:
    """Words for one numeric token in Spanish notation: "1.234,56", "12,5", "1.5", "300".

    ``feminine``/``apocopate`` agree an integer with the noun that follows;
    decimals never apocopate ("veintiuno coma cinco grados").
    """
    token = token.replace(" ", "")
    if "," in token:
        whole, fraction = token.split(",", 1)
        return decimal_words(whole.replace(".", ""), fraction, feminine)
    if "." in token:
        if _is_thousands(token):
            return cardinal(int(token.replace(".", "")), feminine, apocopate)
        whole, fraction = token.split(".", 1)
        return decimal_words(whole, fraction.replace(".", ""), feminine)
    if len(token) >= 10:  # long codes: digit by digit
        return digits(token)
    return cardinal(int(token), feminine, apocopate)


def _amount(token: str) -> tuple[int, str | None]:
    """Split a money token into whole units and the cents string, if any."""
    if "," in token:
        whole, cents = token.split(",", 1)
        return int(whole.replace(".", "") or 0), cents
    if "." in token and not _is_thousands(token):
        whole, cents = token.split(".", 1)
        return int(whole or 0), cents
    return int(token.replace(".", "")), None


def _round_millions(n: int) -> bool:
    return n >= 10**6 and n % 10**6 == 0


def _followed_by_noun(m: re.Match) -> bool:
    """A lowercase word follows that the number most likely counts ("21 días", not "hay 1 disponible")."""
    rest = m.string[m.end() :]
    word = re.match(r"\s+([a-záéíóúñ]+)", rest)
    if not word or word.group(1) in _NOT_NOUNS:
        return False
    return not re.search(r"(?:ible|able)s?$", word.group(1))  # adjectives: the number stands alone


def _sentence_period(m: re.Match) -> str:
    """Keep the period when the abbreviation also ended its sentence."""
    rest = m.string[m.end() :].lstrip(" \t\"'»”’)")  # noqa: RUF001 -- typographic quotes are intended
    if not rest or rest[0] == "\n" or rest[0].isupper() or rest[0] in "¿¡":
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


def _quantity(token: str, unit: str) -> str:
    singular, plural = UNITS_ES[unit.lower()]
    feminine = singular in _FEMININE_NOUNS
    if token in ("1", "1,0", "1,00"):
        return f"{'una' if feminine else 'un'} {singular}"
    return f"{_number_words(token, feminine=feminine, apocopate=not feminine)} {plural}"


def _signed(sign: str, words: str) -> str:
    return f"menos {words}" if sign else words


# --- rules, in priority order --------------------------------------------------------------------


@_rule(r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.]+|#\w+")
def _protect_links(m: re.Match) -> str:
    return m.group(0)


@_rule(
    r"\b\d{4}-\d{2}-\d{2}T\S+"  # ISO 8601 timestamps
    r"|(?<!\d\.)\b\d{1,3}(?:\.\d{1,3}){3}\b(?!\.\d)"  # dotted IP addresses
    r"|\bv\d+(?:\.\d+)+\b|\b(?!\d{1,3}(?:\.\d{3})+\b)\d+\.\d+\.\d+\b"  # version numbers
    r"|\b[A-Za-z]+\d[\w-]*\b|\b[A-Za-z]{2,}-\d+\b"  # Qwen3, GPT-4o, COVID-19, h2o
    r"|\b\d{1,2}/\d{1,2}/\d{2}\b"  # dd/mm/yy: too ambiguous to read
    r"|\b\d+/\d+\b(?![/-]\d)"  # fractions and 24/7
)
def _protect_identifiers(m: re.Match) -> str:
    return m.group(0)


@_rule(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
def _date_dmy(m: re.Match) -> str:
    return _date_words(int(m.group(1)), int(m.group(2)), int(m.group(3))) or m.group(0)


@_rule(r"\b(\d{4})-(\d{2})-(\d{2})\b")
def _date_iso(m: re.Match) -> str:
    return _date_words(int(m.group(3)), int(m.group(2)), int(m.group(1))) or m.group(0)


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
    return f"{cardinal(hour, feminine=True)}{minutes}"  # "la una", "las veintiuna treinta"


_ROMAN_NOUNS = {"vol.": "volumen", "cap.": "capítulo"}


@_rule(r"\b(siglo|capítulo|tomo|parte|volumen|vol\.|cap\.)\s+((?-i:[IVXLC]+))\b")
def _roman_after_noun(m: re.Match) -> str:
    value = _roman_to_int(m.group(2))
    noun = _ROMAN_NOUNS.get(m.group(1).lower(), m.group(1))
    return f"{noun} {cardinal(value)}" if value else m.group(0)


@_rule(rf"(?<![\w.,])(-?)({_NUM})\s?({_UNIT_ALT}){_NOT_IN_WORD}")
def _unit(m: re.Match) -> str:
    return _signed(m.group(1), _quantity(m.group(2), m.group(3)))


@_rule(rf"\b(\d{{1,3}})\.?(º|ª|°|er){_NOT_IN_WORD}")
def _ordinal(m: re.Match) -> str:
    marker = m.group(2).lower()
    return ordinal(int(m.group(1)), feminine=marker == "ª", apocopate=marker == "er")


@_rule(rf"(?<![\w.,])(-?)(?:({_NUM})\s?(€|\$|£|euros?\b|dólar(?:es)?\b|libras?\b)|(?:US)?(€|\$|£)\s?({_NUM}))")
def _money(m: re.Match) -> str:
    token, symbol = (m.group(2), m.group(3)) if m.group(2) else (m.group(5), m.group(4))
    names = CURRENCIES_ES[_CURRENCY_WORDS.get(symbol.lower(), symbol)]
    feminine = names[0] in _FEMININE_NOUNS
    whole, cents = _amount(token)
    if whole == 1:
        amount = "una" if feminine else "un"
    else:
        amount = cardinal(whole, feminine=feminine, apocopate=not feminine)
        if _round_millions(whole):
            amount += " de"
    words = f"{amount} {names[0] if whole == 1 else names[1]}"
    if cents and int(cents):
        value = int(cents.ljust(2, "0")[:2])
        words += f" con {'un' if value == 1 else cardinal(value)} {names[2] if value == 1 else names[3]}"
    return _signed(m.group(1), words)


@_rule(rf"(?<![\w.,])(-?)({_NUM})\s?%")
def _percent(m: re.Match) -> str:
    return _signed(m.group(1), f"{_number_words(m.group(2))} por ciento")


@_rule(r"\b\d{2,4}(?:-\d{2,4}){2,}\b|\b\d{3}(?: \d{3}){2,}\b|\b\d{3}(?: \d{2}){3}\b|\b[6-9]\d{8}\b")
def _digit_groups(m: re.Match) -> str:
    """Phone numbers (912 345 678, 912-345-678, 612 34 56 78, 612345678) read group by group."""
    token = m.group(0)
    groups = [token[i : i + 3] for i in range(0, 9, 3)] if token.isdigit() else re.split(r"[- ]", token)
    return ", ".join(cardinal(int(g)) for g in groups)


@_rule(rf"(?<![\d.,-])(\d{{1,4}})-(\d{{1,4}})(?![\d-]|[.,]\d)(?:\s?({_UNIT_ALT}){_NOT_IN_WORD})?")
def _range(m: re.Match) -> str:
    low, high, unit = m.group(1), m.group(2), m.group(3)
    high_words = _quantity(high, unit) if unit else cardinal(int(high))
    return f"{cardinal(int(low))} a {high_words}"


@_rule(rf"(?<![\w.,])(-?)({_NUM})(?![\w])")
def _number(m: re.Match) -> str:
    token = m.group(2)
    before_noun = _followed_by_noun(m)
    words = _number_words(token, apocopate=before_noun)  # "veintiún días", "un millón"
    plain = token.replace(".", "")
    if before_noun and plain.isdigit() and _round_millions(int(plain)):
        words += " de"  # "dos millones de habitantes"
    return _signed(m.group(1), words)


@_rule(r"\b(?-i:EE)\.\s?(?-i:UU)\.")
def _estados_unidos(m: re.Match) -> str:
    return "Estados Unidos" + _sentence_period(m)


@_rule(r"\bp\.\s?ej\.")
def _por_ejemplo(m: re.Match) -> str:
    return "por ejemplo" + _sentence_period(m)


# "D. Juan" -> "don Juan": only at the start of a phrase, never after another
# initial ("J. D. Salinger"), a lowercase word ("vitamina D.") or a symbol ("I+D.").
@_rule(r"(?<![^\s\"'«“(¿¡])(?<![a-záéíóúñ] )(?<![A-Z]\. )(?-i:D)\.(?=\s+[A-ZÁÉÍÓÚ][a-záéíóúñ])")
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
