"""Spanish-aware text chunking.

The splitter only knew English abbreviations, treated any period after a
digit as a decimal (so "en 1990. Luego" never split), ignored closing
quotes after a sentence end (so «¿Vienes?» never split), and had no notion
of paragraphs, so a blank line could sit inside a chunk and reach the model
verbatim. It also produced 3-word tail chunks that Qwen renders with odd
prosody. The Spanish additions must not break English text.
"""

import numpy as np
import pytest

from backend.utils.chunked_tts import (
    _find_last_sentence_end,
    _merge_tiny_chunks,
    _split_paragraphs,
    generate_chunked,
    split_text_into_chunks,
)

LONG = "Esta es una frase bastante larga que sirve de relleno para las pruebas de partición del texto."


@pytest.mark.parametrize(
    "text",
    [
        "La Dra. Pérez",
        "la Sra. García",
        "la Srta. Ruiz",
        "el núm. 3",
        "Ud. sabe",
        "en la pág. 12",
        "la Avda. Diagonal",
        "en EE. UU. viven",
        "J. R. R. Tolkien",
        "D. Manuel",
        "frutas, p. ej. manzanas",
        "el art. 5 dice",
        "en el cap. 3 y la fig. 2",
        "el vol. IV",
        "Pasos: 1. Abrir",
        "Vale 3.5 euros",
    ],
    ids=["dra", "sra", "srta", "num", "ud", "pag", "avda", "ee-uu", "initials", "don", "p-ej", "art-n", "cap-fig-n", "vol-roman", "list-marker", "decimal"],
)
def test_not_a_sentence_end(text):
    assert _find_last_sentence_end(text) == -1


@pytest.mark.parametrize(
    "text, marker",
    [
        ("Dije que no. Punto", "no."),
        ("Nació en 1990. Luego", "1990."),
        ("Vale 3.5 euros. Fin", "euros."),
        ("It is a work of art. Then we left", "art."),
        ("He wore a red cap. Then", "cap."),
        ("So did I. Then we left", "I."),
        ("Elegimos el plan A. Luego", "A."),
        ("Toma vitamina C. Luego", "C."),
        ("La vi en 3D. Luego", "3D."),
        ("«¿Vienes?» Claro", "?»"),
        ('Dijo "vale". Luego', '".'),
        ("(¿Vienes?) Claro", "?)"),
        ("Espera… Ya voy", "…"),
        ("你好。再见。", "。再见。"),
    ],
    ids=["no", "year", "decimal-then-end", "en-art", "en-cap", "en-i", "plan-a", "vitamina-c", "3d", "guillemets", "quotes", "parens", "ellipsis", "cjk"],
)
def test_sentence_end_lands_on_the_last_char(text, marker):
    expected = text.index(marker) + len(marker) - 1
    if marker == "。再见。":
        expected = len(text) - 1
    assert _find_last_sentence_end(text) == expected


def test_paragraph_breaks_are_chunk_boundaries_even_when_the_text_fits():
    assert split_text_into_chunks(f"{LONG}\n\n{LONG}", 800) == [LONG, LONG]


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_single_newlines_are_collapsed_and_crlf_paragraphs_split(newline):
    assert _split_paragraphs(f"A{newline}B{newline}{newline}C") == ["A B", "C"]


def test_tiny_dialogue_paragraphs_are_folded_into_their_neighbours():
    text = "—¿Vienes conmigo esta tarde? —preguntó ella.\n\n—Sí.\n\n—Vale, pues nos vemos."

    assert split_text_into_chunks(text, 800) == ["—¿Vienes conmigo esta tarde? —preguntó ella. —Sí. —Vale, pues nos vemos."]


def test_list_markers_lead_their_item():
    text = "Pasos a seguir: 1. Abrir la caja con cuidado. 2. Sacar el cable."

    chunks = split_text_into_chunks(text, 50, merge_tiny=False)

    assert chunks[-1] == "2. Sacar el cable."


def test_tiny_tail_chunk_is_merged_when_it_fits():
    sentence = "x" * 89 + "."
    assert split_text_into_chunks(f"{sentence} Sí.", 100) == [f"{sentence} Sí."]


def test_tiny_tail_chunk_stays_separate_when_it_does_not_fit():
    sentence = "x" * 89 + "."
    assert split_text_into_chunks(f"{sentence} Sí.", 92) == [sentence, "Sí."]


def test_tiny_middle_chunk_merges_forward_when_backward_does_not_fit():
    assert _merge_tiny_chunks(["x" * 95, "Sí.", "y" * 50], 97) == ["x" * 95, f"Sí. {'y' * 50}"]


def test_tiny_leading_chunk_is_merged_into_the_next():
    chunks = split_text_into_chunks(f"Hola. {LONG} {LONG}", 120)

    assert chunks[0].startswith("Hola. Esta es una frase")
    assert all(len(c) >= 40 for c in chunks)


def test_runaway_retry_can_opt_out_of_merging():
    text = "A" * 94 + ".   Sí."

    assert len(split_text_into_chunks(text, 100, merge_tiny=False)) == 2


def test_existing_english_behaviour_is_preserved():
    text = f"{'A' * 119}. {'B' * 119}."
    assert split_text_into_chunks(text, 150) == [f"{'A' * 119}.", f"{'B' * 119}."]


@pytest.mark.asyncio
async def test_single_chunk_fast_path_never_sends_a_raw_newline():
    seen = []

    class Backend:
        async def generate(self, text, *_args):
            seen.append(text)
            return np.zeros(100, dtype=np.float32), 1000

    await generate_chunked(Backend(), "Primera línea\nsegunda línea.", {}, max_chunk_chars=800)

    assert seen == ["Primera línea segunda línea."]
