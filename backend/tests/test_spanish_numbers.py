"""Spanish number words (backend/utils/spanish_numbers.py).

Qwen3-TTS misreads digits in Spanish ("1.234,56" came out as "un millón
treinta y cuatro mil quinientos"), so text is verbalized before synthesis.
These tests pin the number core: cardinals, ordinals and decimals the way
they are said in Spain.
"""

import pytest

from backend.utils.spanish_numbers import cardinal, decimal_words, ordinal


@pytest.mark.parametrize(
    ("value", "words"),
    [
        (0, "cero"),
        (1, "uno"),
        (7, "siete"),
        (15, "quince"),
        (16, "dieciséis"),
        (21, "veintiuno"),
        (22, "veintidós"),
        (23, "veintitrés"),
        (26, "veintiséis"),
        (30, "treinta"),
        (31, "treinta y uno"),
        (99, "noventa y nueve"),
        (100, "cien"),
        (101, "ciento uno"),
        (115, "ciento quince"),
        (200, "doscientos"),
        (500, "quinientos"),
        (700, "setecientos"),
        (900, "novecientos"),
        (999, "novecientos noventa y nueve"),
        (1000, "mil"),
        (1001, "mil uno"),
        (1234, "mil doscientos treinta y cuatro"),
        (2026, "dos mil veintiséis"),
        (21000, "veintiún mil"),
        (31000, "treinta y un mil"),
        (100000, "cien mil"),
        (101000, "ciento un mil"),
        (1000000, "un millón"),
        (1000001, "un millón uno"),
        (2000000, "dos millones"),
        (1034500, "un millón treinta y cuatro mil quinientos"),
        (1234567, "un millón doscientos treinta y cuatro mil quinientos sesenta y siete"),
        (1000000000, "mil millones"),
        (1000000000000, "un billón"),
    ],
)
def test_cardinals(value, words):
    assert cardinal(value) == words


def test_numbers_beyond_the_scale_are_read_digit_by_digit():
    assert cardinal(10**18) == "uno" + " cero" * 18
    assert cardinal(999_999_999_999_999_999).startswith("novecientos noventa y nueve mil novecientos")


def test_feminine_cardinals_agree():
    assert cardinal(1, feminine=True) == "una"
    assert cardinal(21, feminine=True) == "veintiuna"
    assert cardinal(200, feminine=True) == "doscientas"
    assert cardinal(1200, feminine=True) == "mil doscientas"
    assert cardinal(21000, feminine=True) == "veintiún mil"  # "mil" is masculine


@pytest.mark.parametrize(
    ("value", "words"),
    [
        (1, "primero"),
        (2, "segundo"),
        (3, "tercero"),
        (7, "séptimo"),
        (9, "noveno"),
        (10, "décimo"),
        (11, "undécimo"),
        (12, "duodécimo"),
        (13, "decimotercero"),
        (19, "decimonoveno"),
        (20, "vigésimo"),
        (21, "vigésimo primero"),
        (30, "trigésimo"),
        (45, "cuadragésimo quinto"),
        (99, "nonagésimo noveno"),
        (100, "centésimo"),
    ],
)
def test_ordinals(value, words):
    assert ordinal(value) == words


def test_feminine_and_apocopated_ordinals():
    assert ordinal(1, feminine=True) == "primera"
    assert ordinal(3, feminine=True) == "tercera"
    assert ordinal(21, feminine=True) == "vigésima primera"
    assert ordinal(1, apocopate=True) == "primer"
    assert ordinal(3, apocopate=True) == "tercer"


def test_ordinals_above_one_hundred_fall_back_to_cardinals():
    assert ordinal(101) == "ciento uno"


@pytest.mark.parametrize(
    ("integer", "fraction", "words"),
    [
        ("12", "5", "doce coma cinco"),
        ("1", "5", "uno coma cinco"),
        ("0", "05", "cero coma cero cinco"),
        ("1234", "56", "mil doscientos treinta y cuatro coma cincuenta y seis"),
        ("3", "14159", "tres coma uno cuatro uno cinco nueve"),
    ],
)
def test_decimals_read_short_fractions_as_numbers_and_long_ones_digit_by_digit(integer, fraction, words):
    assert decimal_words(integer, fraction) == words
