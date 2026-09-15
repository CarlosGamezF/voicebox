"""Pre-synthesis verbalization (backend/utils/verbalize.py, services.generation.text_to_speak).

Whisper measurements on Spanish clones showed the model misreading digits,
dates, money and abbreviations while plain prose came out perfect, so the
text handed to the model is spelled out first. The stored generation text is
untouched; only what the model says changes.
"""

import pytest

from backend.services.generation import text_to_speak
from backend.utils.verbalize import verbalize

# The exact sentences that failed in the evaluation harness.
MEASURED_FAILURES = [
    (
        "La factura ascendió a 1.234,56 € y se pagó el 15/09/2026 con un descuento del 12,5 % por pronto pago, según consta en el recibo n.º 48.",
        "La factura ascendió a mil doscientos treinta y cuatro euros con cincuenta y seis céntimos y se pagó el quince de septiembre de dos mil veintiséis con un descuento del doce coma cinco por ciento por pronto pago, según consta en el recibo número cuarenta y ocho.",
    ),
    (
        "La Dra. Pérez, núm. 3 de la lista, recomendó, p. ej., caminar veinte minutos al día. El Sr. Ortega y la Srta. Ruiz estuvieron de acuerdo.",
        "La doctora Pérez, número tres de la lista, recomendó, por ejemplo, caminar veinte minutos al día. El señor Ortega y la señorita Ruiz estuvieron de acuerdo.",
    ),
]


# Golden cases proposed by three reviewers and vetted by a Spanish-language skeptic (2026-09-15).
GOLDEN_CASES = [
    ("Quedan 0 plazas.", "Quedan cero plazas."),
    ("Solo hay 1 disponible.", "Solo hay uno disponible."),
    ("Tiene 16 años.", "Tiene dieciséis años."),
    ("La respuesta es 21.", "La respuesta es veintiuno."),
    ("De 100 pasamos a 101.", "De cien pasamos a ciento uno."),
    ("Cuesta 1000 y antes costaba 1001.", "Cuesta mil y antes costaba mil uno."),
    ("El año 2026 será clave.", "El año dos mil veintiséis será clave."),
    ("Asistieron 1.234 espectadores.", "Asistieron mil doscientos treinta y cuatro espectadores."),
    (
        "La ciudad tiene 1.234.567 habitantes.",
        "La ciudad tiene un millón doscientos treinta y cuatro mil quinientos sesenta y siete habitantes.",
    ),
    ("Ya son 1.000.000 de descargas.", "Ya son un millón de descargas."),
    ("Mide 1.5 metros.", "Mide uno coma cinco metros."),
    ("Pi es 3,14159 aproximadamente.", "Pi es tres coma uno cuatro uno cinco nueve aproximadamente."),
    ("Subió 12,5 puntos.", "Subió doce coma cinco puntos."),
    ("Un error de 0,05 es aceptable.", "Un error de cero coma cero cinco es aceptable."),
    ("La mínima fue de -5 grados.", "La mínima fue de menos cinco grados."),
    ("El saldo es -12,5.", "El saldo es menos doce coma cinco."),
    ("Creció un 12,5 % este año.", "Creció un doce coma cinco por ciento este año."),
    ("Garantía del 100 %.", "Garantía del cien por ciento."),
    (
        "El total asciende a 1.234,56 €.",
        "El total asciende a mil doscientos treinta y cuatro euros con cincuenta y seis céntimos.",
    ),
    ("Precio: € 20 por persona.", "Precio: veinte euros por persona."),
    ("Cuesta 20 euros.", "Cuesta veinte euros."),
    ("Son $5 la unidad.", "Son cinco dólares la unidad."),
    ("Son 5 $ la unidad.", "Son cinco dólares la unidad."),
    ("Cuesta £3 en Londres.", "Cuesta tres libras en Londres."),
    ("Solo 1 €.", "Solo un euro."),
    ("Cuesta 1,50 €.", "Cuesta un euro con cincuenta céntimos."),
    ("Cuesta 2,05 €.", "Cuesta dos euros con cinco céntimos."),
    ("Recaudó 1.000.000 €.", "Recaudó un millón de euros."),
    (
        "El plan Premium cuesta 9,99 € al mes.",
        "El plan Premium cuesta nueve euros con noventa y nueve céntimos al mes.",
    ),
    ("Llama al 612 345 678.", "Llama al seiscientos doce trescientos cuarenta y cinco seiscientos setenta y ocho."),
    ("Tardará 10-15 minutos.", "Tardará diez a quince minutos."),
    ("Tardará de 10 a 15 minutos.", "Tardará de diez a quince minutos."),
    (
        "Nació en 1999 y se graduó en 2026.",
        "Nació en mil novecientos noventa y nueve y se graduó en dos mil veintiséis.",
    ),
    ("Pesa 5kg.", "Pesa cinco kilos."),
    ("Vive en el 1.º y ella en la 2.ª.", "Vive en el primero y ella en la segunda."),
    ("Fue su 1.er intento.", "Fue su primer intento."),
    ("Quedó 3º en la carrera y ella 10ª.", "Quedó tercero en la carrera y ella décima."),
    ("Consulta el n.º 3.", "Consulta el número tres."),
    (
        "Precio: 20 €.\n\nEnvío: 5 €.\nTotal: 25 €.",
        "Precio: veinte euros.\n\nEnvío: cinco euros.\nTotal: veinticinco euros.",
    ),
]


@pytest.mark.parametrize(("text", "spoken"), GOLDEN_CASES)
def test_golden_cases(text, spoken):
    assert verbalize(text, "es") == spoken


@pytest.mark.parametrize(("text", "spoken"), MEASURED_FAILURES)
def test_the_sentences_that_failed_in_the_harness_are_spelled_out(text, spoken):
    assert verbalize(text, "es") == spoken


def test_other_languages_pass_through_unchanged():
    text = "15 people came on 15/09/2026 and paid 1,234.56 $."
    assert verbalize(text, "en") == text
    assert verbalize(text, None) == text


def test_punctuation_line_breaks_and_casing_survive():
    text = "15 personas vinieron. Llegaron 15.\n\n«Son las 14:30 h», dijo. ¿Quedan 3?"
    assert (
        verbalize(text, "es")
        == "Quince personas vinieron. Llegaron quince.\n\n«Son las catorce treinta», dijo. ¿Quedan tres?"
    )


def test_identifiers_urls_and_versions_are_left_alone():
    text = "Qwen3-TTS v0.6.0, COVID-19, 24/7, GPT-4o, https://example.com/a?b=1, ana.perez@example.com, #tema3, 3D"
    assert verbalize(text, "es") == text


def test_text_to_speak_only_changes_the_words_the_model_hears():
    assert text_to_speak("Llegaron 15.", "es") == "Llegaron quince."
    assert text_to_speak("Llegaron 15.", "es", verbalize=False) == "Llegaron 15."
    assert text_to_speak("15 arrived.", "en") == "15 arrived."
