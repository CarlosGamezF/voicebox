"""Spanish-aware text chunking.

The splitter only knew English abbreviations, treated any period after a
digit as a decimal (so "en 1990. Luego" never split), ignored closing
quotes after a sentence end (so «¿Vienes?» never split), and had no notion
of paragraphs, so a "\\n\\n" could sit inside a chunk and reach the model
verbatim. It also produced 3-word tail chunks that Qwen renders with odd
prosody.
"""

from backend.utils.chunked_tts import _find_last_sentence_end, split_text_into_chunks

LONG = "Esta es una frase bastante larga que sirve de relleno para las pruebas de partición del texto."


def test_spanish_abbreviations_are_not_sentence_ends():
    assert _find_last_sentence_end("La Dra. Pérez") == -1
    assert _find_last_sentence_end("la Sra. García") == -1
    assert _find_last_sentence_end("la Srta. Ruiz") == -1
    assert _find_last_sentence_end("el núm. 3") == -1
    assert _find_last_sentence_end("Ud. sabe") == -1
    assert _find_last_sentence_end("en la pág. 12") == -1
    assert _find_last_sentence_end("la Avda. Diagonal") == -1


def test_initials_and_p_ej_are_not_sentence_ends():
    assert _find_last_sentence_end("J. R. R. Tolkien") == -1
    assert _find_last_sentence_end("D. Manuel") == -1
    assert _find_last_sentence_end("frutas, p. ej. manzanas") == -1


def test_real_sentence_ends_still_count():
    text = "Dije que no. Punto"
    assert text[_find_last_sentence_end(text)] == "."
    assert _find_last_sentence_end("Dije que no.") == len("Dije que no.") - 1


def test_sentence_end_after_a_number_is_a_boundary():
    text = "Nació en 1990. Luego"
    assert _find_last_sentence_end(text) == text.index(". ")


def test_decimal_numbers_are_not_boundaries():
    text = "Vale 3.5 euros. Fin"
    assert _find_last_sentence_end(text) == text.index("euros.") + len("euros")


def test_closing_quotes_belong_to_the_sentence():
    text = "«¿Vienes?» Claro"
    assert _find_last_sentence_end(text) == text.index("»")
    text = "Dijo \"vale\". Luego"
    assert _find_last_sentence_end(text) == text.index('". ') + 1


def test_ellipsis_ends_a_sentence():
    text = "Espera… Ya voy"
    assert _find_last_sentence_end(text) == text.index("…")


def test_paragraph_breaks_are_chunk_boundaries_even_when_the_text_fits():
    text = f"{LONG}\n\n{LONG}"

    assert split_text_into_chunks(text, 800) == [LONG, LONG]


def test_single_newlines_are_collapsed_to_spaces():
    text = "Primera línea\nsegunda línea del mismo párrafo."

    assert split_text_into_chunks(text, 800) == ["Primera línea segunda línea del mismo párrafo."]


def test_tiny_tail_chunk_is_merged_when_it_fits():
    sentence = "x" * 89 + "."
    text = f"{sentence} Sí."

    assert split_text_into_chunks(text, 100) == [f"{sentence} Sí."]


def test_tiny_tail_chunk_stays_separate_when_it_does_not_fit():
    sentence = "x" * 89 + "."
    text = f"{sentence} Sí."

    assert split_text_into_chunks(text, 92) == [sentence, "Sí."]


def test_tiny_leading_chunk_is_merged_into_the_next():
    text = f"Hola. {LONG} {LONG}"

    chunks = split_text_into_chunks(text, 120)

    assert chunks[0].startswith("Hola. Esta es una frase")
    assert all(len(c) >= 40 for c in chunks)


def test_existing_english_behaviour_is_preserved():
    text = f"{'A' * 119}. {'B' * 119}."

    assert split_text_into_chunks(text, 150) == [f"{'A' * 119}.", f"{'B' * 119}."]
