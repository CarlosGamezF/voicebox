"""Number words as said in Spain: cardinals, ordinals and decimals.

The TTS model reads digits unreliably in Spanish, so the verbalizer spells
numbers out before synthesis. Only the forms needed for that are here;
agreement with a following noun is the caller's decision (``feminine``).
"""

_UNITS = [
    "cero",
    "uno",
    "dos",
    "tres",
    "cuatro",
    "cinco",
    "seis",
    "siete",
    "ocho",
    "nueve",
    "diez",
    "once",
    "doce",
    "trece",
    "catorce",
    "quince",
    "dieciséis",
    "diecisiete",
    "dieciocho",
    "diecinueve",
    "veinte",
    "veintiuno",
    "veintidós",
    "veintitrés",
    "veinticuatro",
    "veinticinco",
    "veintiséis",
    "veintisiete",
    "veintiocho",
    "veintinueve",
]
_TENS = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta", 8: "ochenta", 9: "noventa"}
_HUNDREDS = {
    1: "ciento",
    2: "doscientos",
    3: "trescientos",
    4: "cuatrocientos",
    5: "quinientos",
    6: "seiscientos",
    7: "setecientos",
    8: "ochocientos",
    9: "novecientos",
}
_ORDINAL_UNITS = ["", "primero", "segundo", "tercero", "cuarto", "quinto", "sexto", "séptimo", "octavo", "noveno"]
_ORDINAL_TENS = {
    1: "décimo",
    2: "vigésimo",
    3: "trigésimo",
    4: "cuadragésimo",
    5: "quincuagésimo",
    6: "sexagésimo",
    7: "septuagésimo",
    8: "octogésimo",
    9: "nonagésimo",
}
_ORDINAL_TEENS = {11: "undécimo", 12: "duodécimo", 17: "decimoséptimo", 18: "decimoctavo"}


def cardinal(n: int, feminine: bool = False, apocopate: bool = False) -> str:
    """``1234`` -> ``"mil doscientos treinta y cuatro"``.

    ``feminine`` gives ``"una"``/``"doscientas"``; ``apocopate`` gives the form
    used before a masculine noun (``"veintiún grados"``, ``"un euro"``).
    """
    if n < 0:
        return "menos " + cardinal(-n, feminine, apocopate)
    if n == 0:
        return "cero"
    if n >= 10**18:  # beyond "billones": read digit by digit rather than fail
        return digits(str(n))
    parts: list[str] = []
    billions, rest = divmod(n, 10**12)
    if billions:
        parts.append("un billón" if billions == 1 else f"{_group(billions, apocopate=True)} billones")
    millions, rest = divmod(rest, 10**6)
    if millions:
        parts.append("un millón" if millions == 1 else f"{_group(millions, apocopate=True)} millones")
    if rest:
        parts.append(_group(rest, feminine=feminine, apocopate=apocopate))
    return " ".join(parts)


def ordinal(n: int, feminine: bool = False, apocopate: bool = False) -> str:
    """``3`` -> ``"tercero"``/``"tercera"``/``"tercer"``; above 100 the cardinal is used instead."""
    if n < 1 or n > 100:
        return cardinal(n, feminine)
    if n == 100:
        words = "centésimo"
    elif n in _ORDINAL_TEENS:
        words = _ORDINAL_TEENS[n]
    elif 13 <= n <= 19:
        words = "decimo" + _ORDINAL_UNITS[n - 10]
    else:
        tens, units = divmod(n, 10)
        words = " ".join(w for w in (_ORDINAL_TENS.get(tens, ""), _ORDINAL_UNITS[units]) if w)
    if apocopate and n % 10 in (1, 3) and n % 100 != 11 and n % 100 != 13:
        words = words[:-1]  # primero -> primer, tercero -> tercer
    elif feminine:
        words = " ".join(w[:-1] + "a" if w.endswith("o") else w for w in words.split())
    return words


def decimal_words(integer: str, fraction: str, feminine: bool = False) -> str:
    """``("12", "5")`` -> ``"doce coma cinco"``; fractions longer than two digits are read digit by digit."""
    whole = cardinal(int(integer), feminine)
    if len(fraction) <= 2:
        part = cardinal(int(fraction))
        if fraction.startswith("0") and len(fraction) == 2:
            part = f"cero {part}"
    else:
        part = digits(fraction)
    return f"{whole} coma {part}"


def digits(text: str) -> str:
    """Read a digit string one digit at a time: ``"612"`` -> ``"seis uno dos"``."""
    return " ".join(_UNITS[int(d)] for d in text if d.isdigit())


def _group(n: int, feminine: bool = False, apocopate: bool = False) -> str:
    """Words for 1..999999: the thousands group is always masculine and apocopated ("veintiún mil")."""
    thousands, units = divmod(n, 1000)
    parts: list[str] = []
    if thousands == 1:
        parts.append("mil")
    elif thousands:
        parts.append(f"{_below_thousand(thousands, apocopate=True)} mil")
    if units:
        parts.append(_below_thousand(units, feminine=feminine, apocopate=apocopate))
    return " ".join(parts)


def _below_thousand(n: int, feminine: bool = False, apocopate: bool = False) -> str:
    if n == 100:
        return "cien"
    hundreds, rest = divmod(n, 100)
    parts: list[str] = []
    if hundreds:
        word = _HUNDREDS[hundreds]
        parts.append(word[:-2] + "as" if feminine and hundreds > 1 else word)
    if rest:
        parts.append(_below_hundred(rest, feminine, apocopate))
    return " ".join(parts)


def _below_hundred(n: int, feminine: bool, apocopate: bool) -> str:
    if n < 30:
        word = _UNITS[n]
        if n in (1, 21):
            if apocopate:
                word = "un" if n == 1 else "veintiún"
            elif feminine:
                word = "una" if n == 1 else "veintiuna"
        return word
    tens, units = divmod(n, 10)
    if not units:
        return _TENS[tens]
    return f"{_TENS[tens]} y {_below_hundred(units, feminine, apocopate)}"
