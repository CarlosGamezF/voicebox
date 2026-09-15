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
    # Second panel: dates, times, abbreviations, units, ordinals and text that must not change.
    (
        "La reunión es el 15/09/2026 por la mañana.",
        "La reunión es el quince de septiembre de dos mil veintiséis por la mañana.",
    ),
    ("Firmaron el contrato el 03-11-2024.", "Firmaron el contrato el tres de noviembre de dos mil veinticuatro."),
    ("Fecha de entrega: 2026-01-07.", "Fecha de entrega: siete de enero de dos mil veintiséis."),
    ("El código 13/45/2026 no es una fecha válida.", "El código 13/45/2026 no es una fecha válida."),
    ("El tren sale a las 14:30 desde Atocha.", "El tren sale a las catorce treinta desde Atocha."),
    ("Abrimos a las 9:05 cada día.", "Abrimos a las nueve cero cinco cada día."),
    ("Nos vemos a las 14:30 h en la puerta.", "Nos vemos a las catorce treinta en la puerta."),
    ("El contador vuelve a 00:00 cada noche.", "El contador vuelve a cero cada noche."),
    (
        "Nací en 1999 y me mudé a Madrid en 2026.",
        "Nací en mil novecientos noventa y nueve y me mudé a Madrid en dos mil veintiséis.",
    ),
    ("La cita es el 01/02/2026 a las 8:00.", "La cita es el uno de febrero de dos mil veintiséis a las ocho."),
    ("15/09/2026 es la fecha límite.", "Quince de septiembre de dos mil veintiséis es la fecha límite."),
    ("Sr. Ortega, pase a la sala.", "Señor Ortega, pase a la sala."),
    ("Ha llegado el Sr. Ortega con la Dra. Ruiz.", "Ha llegado el señor Ortega con la doctora Ruiz."),
    ("D. Juan y Dña. Ana firmaron el acuerdo.", "Don Juan y doña Ana firmaron el acuerdo."),
    ("¿Vienen Uds. a la reunión?", "¿Vienen ustedes a la reunión?"),
    ("Hay frutas, p. ej., manzanas y peras.", "Hay frutas, por ejemplo, manzanas y peras."),
    ("Trajo pan, leche, huevos, etc. Luego se fue.", "Trajo pan, leche, huevos, etcétera. Luego se fue."),
    ("Nació en EE. UU. Su madre es de Cádiz.", "Nació en Estados Unidos. Su madre es de Cádiz."),
    ("Véase la pág. 4 y el art. 12.", "Véase la página cuatro y el artículo doce."),
    ("La tienda está en la c/ Mayor 5.", "La tienda está en la calle Mayor cinco."),
    (
        "Llame al tel. 912 345 678.",
        "Llame al teléfono novecientos doce, trescientos cuarenta y cinco, seiscientos setenta y ocho.",
    ),
    (
        "Somos aprox. 20 personas en el dpto. de ventas.",
        "Somos aproximadamente veinte personas en el departamento de ventas.",
    ),
    ("Domina el art de vivir sin prisa.", "Domina el art de vivir sin prisa."),
    ("Queda 1 km hasta el pueblo y luego 5km más.", "Queda un kilómetro hasta el pueblo y luego cinco kilómetros más."),
    ("El paquete pesa 12,5 kg.", "El paquete pesa doce coma cinco kilos."),
    (
        "El viaje dura 3 h y 45 min, con paradas de 10 s.",
        "El viaje dura tres horas y cuarenta y cinco minutos, con paradas de diez segundos.",
    ),
    ("Iba a 120 km/h cuando le pararon.", "Iba a ciento veinte kilómetros por hora cuando le pararon."),
    ("Hoy hace 21 °C en Sevilla.", "Hoy hace veintiún grados en Sevilla."),
    ("Esta noche bajaremos a -5 °C en Burgos.", "Esta noche bajaremos a menos cinco grados en Burgos."),
    ("Vive en el 3.º y su hermana en el 1.er piso.", "Vive en el tercero y su hermana en el primer piso."),
    ("Quedó en 1.ª posición y su prima en la 10ª.", "Quedó en primera posición y su prima en la décima."),
    ("El siglo XIX fue el de la Revolución industrial.", "El siglo diecinueve fue el de la Revolución industrial."),
    ("Véase el cap. IV, pág. 12.", "Véase el capítulo cuatro, página doce."),
    ("Deja reposar 10-15 s antes de hornear.", "Deja reposar diez a quince segundos antes de hornear."),
    ("La feria será el 3-5 de mayo.", "La feria será el tres a cinco de mayo."),
    ("El motor Qwen3-TTS lee bien.", "El motor Qwen3-TTS lee bien."),
    ("Probamos GPT-4o ayer.", "Probamos GPT-4o ayer."),
    ("La versión v0.6.0 salió durante la COVID-19.", "La versión v0.6.0 salió durante la COVID-19."),
    ("Atención 24/7 y añade 1/2 taza de leche.", "Atención 24/7 y añade 1/2 taza de leche."),
    (
        "Consulta https://voicebox.dev/docs/v0.6.0 para instalarlo.",
        "Consulta https://voicebox.dev/docs/v0.6.0 para instalarlo.",
    ),
    ("Escribe a ana.lopez2@ejemplo.es con el asunto #tema3.", "Escribe a ana.lopez2@ejemplo.es con el asunto #tema3."),
    ("Modelamos la molécula h2o en 3D para la clase.", "Modelamos la molécula h2o en 3D para la clase."),
    ("Marcó «15» en la casilla.", "Marcó «quince» en la casilla."),
    ("Llegaron 15.", "Llegaron quince."),
    ("Quedan 15\nNada más.\n\nSigue el 2.", "Quedan quince\nNada más.\n\nSigue el dos."),
    ("Juan A. Pérez eligió el plan B. Nadie protestó.", "Juan A. Pérez eligió el plan B. Nadie protestó."),
    ("Vino el Sr Ruiz a la reunión.", "Vino el Sr Ruiz a la reunión."),
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
    # Phone groups are read with pauses: the second panel's convention won over this one's.
    ("Llama al 612 345 678.", "Llama al seiscientos doce, trescientos cuarenta y cinco, seiscientos setenta y ocho."),
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


# Cases from the adversarial review of the first version (2026-09-15).
REVIEW_CASES = [
    ("Cuesta 2.50 €.", "Cuesta dos euros con cincuenta céntimos."),
    ("Cuesta $1.5.", "Cuesta un dólar con cincuenta centavos."),
    ("Vale 1.5 euros.", "Vale un euro con cincuenta céntimos."),
    ("Son €5.99 al mes.", "Son cinco euros con noventa y nueve céntimos al mes."),
    ("Cobra US$ 5.", "Cobra cinco dólares."),
    ("La parte civil del proceso.", "La parte civil del proceso."),
    ("Lo tomo vi.", "Lo tomo vi."),
    ("Toma vitamina D. La dosis es alta.", "Toma vitamina D. La dosis es alta."),
    ("Leí a J. D. Salinger.", "Leí a J. D. Salinger."),
    ("Hablamos de I+D. Luego seguimos.", "Hablamos de I+D. Luego seguimos."),
    ("D. Juan llegó.", "Don Juan llegó."),
    ("Hace 25°C.", "Hace veinticinco grados."),
    ("Vive en el 5° piso.", "Vive en el quinto piso."),
    ("Tardó 1 día.", "Tardó un día."),
    ("Costó 1 millón de euros.", "Costó un millón de euros."),
    ("Faltan 21 días.", "Faltan veintiún días."),
    ("A la 1:30 llegó.", "A la una treinta llegó."),
    ("A las 21:30 h.", "A las veintiuna treinta."),
    ("Fue el 2026-09-15T14:30:00 en 192.168.1.1.", "Fue el 2026-09-15T14:30:00 en 192.168.1.1."),
    (
        "www.ejemplo.com tiene 15 usuarios. v0.6.0 ya está. h2o es agua. hola@ej.com me escribe.",
        "www.ejemplo.com tiene quince usuarios. v0.6.0 ya está. h2o es agua. hola@ej.com me escribe.",
    ),
    ("Llama al 612345678.", "Llama al seiscientos doce, trescientos cuarenta y cinco, seiscientos setenta y ocho."),
    ("Código 1234567890.", "Código uno dos tres cuatro cinco seis siete ocho nueve cero."),
    ("Nació el 15/09/26.", "Nació el 15/09/26."),
    ("Bajó un -5 %.", "Bajó un menos cinco por ciento."),
]


@pytest.mark.parametrize(("text", "spoken"), REVIEW_CASES)
def test_review_cases(text, spoken):
    assert verbalize(text, "es") == spoken


def test_absurdly_large_numbers_degrade_to_digits_instead_of_failing_the_take():
    spoken = verbalize("Hubo 1.000.000.000.000.000.000 de intentos.", "es")
    assert not any(ch.isdigit() for ch in spoken)
    assert spoken.startswith("Hubo uno cero cero")


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
