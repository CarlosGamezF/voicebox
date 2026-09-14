"""Pure parts of the prosody evaluation harness (backend/tools/prosody_eval.py).

The harness drives a running Voicebox backend over REST; these tests cover
the pieces that need no server: text normalisation and WER, the audio
metrics, the request built per condition and the blind pairing.
"""

import numpy as np

from backend.tools.prosody_eval import (
    CORPUS,
    Condition,
    audio_metrics,
    build_generate_payload,
    make_blind_pairs,
    normalize_text,
    wer,
)

SR = 1000


def test_normalize_text_lowercases_strips_punctuation_and_accents():
    assert normalize_text("¿Qué medidas se tomarán? ¡Paciencia!") == "que medidas se tomaran paciencia"
    assert normalize_text("La Dra. Pérez, núm. 3.") == "la doctora perez número 3"


def test_wer_counts_substitutions_insertions_and_deletions():
    assert wer("uno dos tres cuatro cinco", "uno dos tres cuatro cinco") == 0.0
    assert wer("uno dos tres cuatro cinco", "uno dos tres cuatro seis") == 0.2
    assert wer("uno dos tres", "uno dos tres cuatro") == 1 / 3
    assert wer("uno dos tres", "uno tres") == 1 / 3
    assert wer("", "algo") == 1.0


def test_audio_metrics_measure_silence_pauses_and_rate():
    speech = np.full(SR, 0.2, dtype=np.float32)  # 1 s
    pause = np.zeros(int(0.4 * SR), dtype=np.float32)
    audio = np.concatenate([np.zeros(int(0.1 * SR), np.float32), speech, pause, speech, np.zeros(int(0.3 * SR), np.float32)])

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


def test_corpus_covers_the_protocol_cases():
    ids = {item["id"] for item in CORPUS}
    assert {"short_question", "narrative_date", "long_paragraph", "two_paragraphs", "numbers_currency", "abbreviations"} <= ids
    assert all(len(item["text"]) >= 30 for item in CORPUS)
