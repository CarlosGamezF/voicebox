"""MLXTTSBackend.generate(): no silent fallback to the default voice, plus telemetry.

Two code paths used to swap the cloned voice for the model's default voice
and only log a warning: a missing reference file, and any exception raised
by the cloning call. From the UI the result looked like a bad clone with a
foreign accent. Cloning problems must fail the generation instead.

The backend also logs the language actually sent to mlx-audio and warns when
a chunk hits mlx-audio's ICL token cap (max(75, 6 * text tokens)), which
otherwise truncates slow or pause-heavy speech without any signal.
"""

import logging

import numpy as np
import pytest

from backend.backends.mlx_backend import MLXTTSBackend
from backend.utils.mlx_executor import shutdown_mlx_executor


@pytest.fixture(autouse=True)
def _fresh_executor():
    shutdown_mlx_executor()
    yield
    shutdown_mlx_executor()


class _Result:
    def __init__(self, samples: int, token_count: int = 10):
        self.audio = np.zeros(samples, dtype=np.float32)
        self.sample_rate = 24000
        self.token_count = token_count


class _FakeTokenizer:
    def encode(self, text):
        # ~4 characters per token, like Spanish on the Qwen tokenizer.
        return list(range(max(1, len(text) // 4)))


class _FakeModel:
    def __init__(self, results=None, raise_exc=None, has_encoder=True, with_tokenizer=True):
        self.calls = []
        self._results = [_Result(2400)] if results is None else results
        self._raise = raise_exc
        self.speech_tokenizer = type("SpeechTokenizer", (), {"has_encoder": has_encoder})()
        if with_tokenizer:
            self.tokenizer = _FakeTokenizer()

    def generate(self, text, ref_audio=None, ref_text=None, lang_code="auto", **kwargs):
        self.calls.append({"text": text, "ref_audio": ref_audio, "ref_text": ref_text, "lang_code": lang_code})
        if self._raise is not None:
            raise self._raise
        yield from self._results


class _NoCloneModel:
    """A model whose generate() has no ref_audio parameter."""

    def __init__(self):
        self.calls = 0

    def generate(self, text, lang_code="auto"):
        self.calls += 1
        yield _Result(2400)


def _backend(model) -> MLXTTSBackend:
    backend = MLXTTSBackend()
    backend.model = model
    backend._current_model_size = "1.7B"
    return backend


def _prompt(tmp_path, text="hola"):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    return {"ref_audio": str(ref), "ref_text": text}


async def test_missing_reference_file_fails_instead_of_default_voice(tmp_path):
    model = _FakeModel()
    backend = _backend(model)

    with pytest.raises(FileNotFoundError, match="Reference audio"):
        await backend.generate("hola", {"ref_audio": str(tmp_path / "gone.wav"), "ref_text": "hola"}, language="es")

    assert model.calls == [], "the model must not be asked for a default-voice take"


async def test_cloning_error_propagates_without_regenerating(tmp_path):
    model = _FakeModel(raise_exc=RuntimeError("There is no Stream(gpu, 2) in current thread."))
    backend = _backend(model)

    with pytest.raises(RuntimeError, match="Stream"):
        await backend.generate("hola", _prompt(tmp_path), language="es")

    assert len(model.calls) == 1
    assert model.calls[0]["ref_audio"] is not None


async def test_model_without_cloning_support_is_an_error(tmp_path):
    model = _NoCloneModel()
    backend = _backend(model)

    with pytest.raises(RuntimeError, match="does not support voice cloning"):
        await backend.generate("hola", _prompt(tmp_path), language="es")

    assert model.calls == 0


async def test_empty_output_is_an_error(tmp_path):
    backend = _backend(_FakeModel(results=[]))

    with pytest.raises(RuntimeError, match="no audio"):
        await backend.generate("hola", _prompt(tmp_path), language="es")


async def test_clone_passes_reference_transcript_and_spanish(tmp_path):
    model = _FakeModel()
    backend = _backend(model)

    audio, sample_rate = await backend.generate("hola", _prompt(tmp_path, "texto de referencia"), language="es")

    assert sample_rate == 24000
    assert len(audio) == 2400
    call = model.calls[0]
    assert call["lang_code"] == "spanish"
    assert call["ref_text"] == "texto de referencia"


async def test_prompt_without_reference_still_generates_plain():
    model = _FakeModel()
    backend = _backend(model)

    audio, _ = await backend.generate("hola", {}, language="es")

    assert len(audio) == 2400
    assert model.calls[0]["ref_audio"] is None


async def test_unsupported_language_falls_back_to_auto_with_warning(tmp_path, caplog):
    model = _FakeModel()
    backend = _backend(model)

    with caplog.at_level(logging.WARNING):
        await backend.generate("shalom", _prompt(tmp_path), language="he")

    assert model.calls[0]["lang_code"] == "auto"
    assert any("not supported" in r.getMessage() for r in caplog.records)


async def test_hitting_the_token_cap_logs_a_warning(tmp_path, caplog):
    # 40 chars -> 10 text tokens -> cap = max(75, 60) = 75 codec tokens.
    text = "x" * 40
    backend = _backend(_FakeModel(results=[_Result(2400, token_count=75)]))

    with caplog.at_level(logging.WARNING):
        await backend.generate(text, _prompt(tmp_path), language="es")

    assert any("token cap" in r.getMessage() for r in caplog.records)


async def test_below_the_token_cap_stays_quiet(tmp_path, caplog):
    backend = _backend(_FakeModel(results=[_Result(2400, token_count=30)]))

    with caplog.at_level(logging.WARNING):
        await backend.generate("x" * 40, _prompt(tmp_path), language="es")

    assert not any("token cap" in r.getMessage() for r in caplog.records)


async def test_logs_language_and_icl_mode(tmp_path, caplog):
    backend = _backend(_FakeModel())

    with caplog.at_level(logging.INFO):
        await backend.generate("hola", _prompt(tmp_path), language="es")

    messages = [r.getMessage() for r in caplog.records]
    assert any("lang_code=spanish" in m and "icl=True" in m for m in messages)


async def test_legacy_ref_audio_path_key_still_clones(tmp_path):
    model = _FakeModel()
    backend = _backend(model)
    ref = tmp_path / "legacy.wav"
    ref.write_bytes(b"RIFF")

    await backend.generate("hola", {"ref_audio_path": str(ref), "ref_text": "hola"}, language="es")

    assert model.calls[0]["ref_audio"] == str(ref)


async def test_missing_transcript_is_sent_as_empty_string_with_warning(tmp_path, caplog):
    model = _FakeModel()
    backend = _backend(model)
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")

    with caplog.at_level(logging.WARNING):
        await backend.generate("hola", {"ref_audio": str(ref), "ref_text": None}, language="es")

    assert model.calls[0]["ref_text"] == ""
    assert any("transcript is empty" in r.getMessage() for r in caplog.records)


async def test_model_without_tokenizer_skips_the_cap_check(tmp_path, caplog):
    backend = _backend(_FakeModel(results=[_Result(2400, token_count=75)], with_tokenizer=False))

    with caplog.at_level(logging.WARNING):
        audio, _ = await backend.generate("x" * 40, _prompt(tmp_path), language="es")

    assert len(audio) == 2400
    assert not any("token cap" in r.getMessage() for r in caplog.records)


async def test_results_without_token_count_stay_quiet(tmp_path, caplog):
    class _Bare:
        audio = np.zeros(2400, dtype=np.float32)
        sample_rate = 24000

    backend = _backend(_FakeModel(results=[_Bare()]))

    with caplog.at_level(logging.WARNING):
        audio, _ = await backend.generate("x" * 40, _prompt(tmp_path), language="es")

    assert len(audio) == 2400
    assert not any("token cap" in r.getMessage() for r in caplog.records)


async def test_cap_is_checked_per_newline_segment(tmp_path, caplog):
    # Two segments of 40 chars: caps are 75 each. Only the second is capped;
    # summed against the whole text (cap 120) it would have been masked.
    text = "x" * 40 + "\n" + "y" * 40
    backend = _backend(_FakeModel(results=[_Result(2400, token_count=30), _Result(2400, token_count=75)]))

    with caplog.at_level(logging.WARNING):
        await backend.generate(text, _prompt(tmp_path), language="es")

    assert any("token cap" in r.getMessage() for r in caplog.records)


async def test_tokenizer_failure_never_fails_a_take(tmp_path):
    class _BrokenTokenizer:
        def encode(self, text):
            raise ValueError("bad input")

    model = _FakeModel()
    model.tokenizer = _BrokenTokenizer()
    backend = _backend(model)

    audio, _ = await backend.generate("hola", _prompt(tmp_path), language="es")

    assert len(audio) == 2400
