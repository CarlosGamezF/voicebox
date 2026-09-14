"""Pure parts of the prosody evaluation harness (backend/tools/prosody_eval.py).

The harness drives a running Voicebox backend over REST; these tests cover
the pieces that need no server: text normalisation and WER, the audio
metrics, the request built per condition and the blind pairing.
"""

import argparse

import numpy as np
import pytest

from backend.tools.prosody_eval import (
    CORPUS,
    Condition,
    audio_metrics,
    build_generate_payload,
    make_blind_pairs,
    normalize_text,
    parse_condition,
    wer,
)

SR = 1000


def test_normalize_text_lowercases_strips_punctuation_and_accents():
    assert normalize_text("¿Qué medidas se tomarán? ¡Paciencia!") == "que medidas se tomaran paciencia"
    assert normalize_text("La Dra. Pérez, núm. 3.") == "la doctora perez numero 3"


def test_normalize_text_expands_abbreviations_the_way_whisper_writes_them():
    # Expansions carry accents; Whisper output is compared accent-free, so they must be stripped too.
    assert normalize_text("núm. 3") == normalize_text("número 3")
    assert normalize_text("Recomendó, p. ej., caminar.") == "recomendo por ejemplo caminar"
    assert normalize_text("El n.º 3 y la pág. 4") == "el numero 3 y la pagina 4"


def test_wer_counts_substitutions_insertions_and_deletions():
    assert wer("uno dos tres cuatro cinco", "uno dos tres cuatro cinco") == 0.0
    assert wer("uno dos tres cuatro cinco", "uno dos tres cuatro seis") == 0.2
    assert wer("uno dos tres", "uno dos tres cuatro") == 1 / 3
    assert wer("uno dos tres", "uno tres") == 1 / 3
    assert wer("", "algo") == 1.0


def test_audio_metrics_measure_silence_pauses_and_rate():
    speech = np.full(SR, 0.2, dtype=np.float32)  # 1 s
    pause = np.zeros(int(0.4 * SR), dtype=np.float32)
    audio = np.concatenate(
        [np.zeros(int(0.1 * SR), np.float32), speech, pause, speech, np.zeros(int(0.3 * SR), np.float32)]
    )

    m = audio_metrics(audio, SR, text="x" * 28)

    assert m["duration_s"] == 2.8
    assert m["leading_silence_s"] == 0.1
    assert abs(m["trailing_silence_s"] - 0.3) < 0.002
    assert m["pauses_over_300ms"] == 1
    assert abs(m["longest_pause_s"] - 0.4) < 0.01
    assert m["chars_per_s"] == 10.0
    assert m["clipped_085_ratio"] == 0.0


def test_build_generate_payload_merges_condition_overrides():
    cond = Condition(name="chunk300", overrides={"max_chunk_chars": 300, "crossfade_ms": 80})

    payload = build_generate_payload(cond, profile_id="p1", text="hola", seed=7, language="es")

    assert payload["profile_id"] == "p1"
    assert payload["seed"] == 7
    assert payload["language"] == "es"
    assert payload["max_chunk_chars"] == 300
    assert payload["crossfade_ms"] == 80
    assert payload["normalize"] is True  # default kept unless overridden


def test_blind_pairs_are_deterministic_and_hide_the_condition():
    rows = [
        {"text_id": "t1", "seed": 1, "condition": "a", "wav": "a1.wav"},
        {"text_id": "t1", "seed": 1, "condition": "b", "wav": "b1.wav"},
        {"text_id": "t2", "seed": 1, "condition": "a", "wav": "a2.wav"},
        {"text_id": "t2", "seed": 1, "condition": "b", "wav": "b2.wav"},
    ]

    pairs, key = make_blind_pairs(rows, rng_seed=3)
    pairs_again, _ = make_blind_pairs(rows, rng_seed=3)

    assert pairs == pairs_again
    assert len(pairs) == 2
    for pair in pairs:
        assert set(pair) == {"pair_id", "text_id", "seed", "A", "B"}
        assert {pair["A"], pair["B"]} == {"a1.wav", "b1.wav"} or {pair["A"], pair["B"]} == {"a2.wav", "b2.wav"}
    assert set(key[pairs[0]["pair_id"]].values()) == {"a", "b"}


def test_blind_pairs_skip_failed_takes():
    rows = [
        {"text_id": "t1", "seed": 1, "condition": "a", "wav": "a1.wav", "status": "completed"},
        {"text_id": "t1", "seed": 1, "condition": "b", "status": "failed", "error": "boom"},
        {"text_id": "t2", "seed": 1, "condition": "a", "wav": "a2.wav", "status": "completed"},
        {"text_id": "t2", "seed": 1, "condition": "b", "wav": "b2.wav", "status": "completed"},
    ]

    pairs, key = make_blind_pairs(rows)

    assert [p["text_id"] for p in pairs] == ["t2"]
    assert set(key[pairs[0]["pair_id"]].values()) == {"a", "b"}


def test_parse_condition_coerces_ints_floats_and_booleans():
    cond = parse_condition("c:max_chunk_chars=300,crossfade_ms=80.5,normalize=false,language=es")

    assert cond.name == "c"
    assert cond.overrides == {"max_chunk_chars": 300, "crossfade_ms": 80.5, "normalize": False, "language": "es"}
    assert parse_condition("bare").overrides == {}


@pytest.mark.parametrize("spec", ["x:a", ":a=1", "x:=1", "x:a=nan"])
def test_parse_condition_rejects_malformed_specs(spec):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_condition(spec)


def test_corpus_covers_the_protocol_cases():
    ids = {item["id"] for item in CORPUS}
    assert {
        "short_question",
        "narrative_date",
        "long_paragraph",
        "two_paragraphs",
        "numbers_currency",
        "abbreviations",
    } <= ids
    assert all(len(item["text"]) >= 30 for item in CORPUS)
