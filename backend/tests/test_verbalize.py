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
