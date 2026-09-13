"""The default Whisper size follows the persisted STT setting when it is usable.

``POST /transcribe`` (voice-sample "Transcribe" button, REST callers) and the
MCP transcribe tool used to fall back to whatever size the STT backend held,
i.e. ``base`` on a cold backend, ignoring the model configured in Settings.
The resolver makes the configured ``stt_model`` the default, but only when
that model is loaded or already downloaded: the 0.5.0 desktop UI treats the
202 "downloading" reply as a success, and MCP would error, so an uncached
setting must degrade to the current size (with a warning), not surprise the
user with a 1.5 GB download.
"""

import logging

import pytest

from backend.mcp_server import tools
from backend.services import transcribe


class _Settings:
    stt_model = "large"


class _FakeWhisper:
    def __init__(self, loaded_size="base", cached=()):
        self.model_size = loaded_size
        self._cached = set(cached)
        self.calls = []

    def is_loaded(self):
        return True

    def _is_model_cached(self, size):
        return size in self._cached

    async def transcribe(self, path, language, model_size):
        self.calls.append(model_size)
        return "texto"


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(transcribe, "get_capture_settings", lambda db: _Settings())


def test_explicit_model_wins():
    whisper = _FakeWhisper(cached={"large"})

    assert transcribe.resolve_transcription_model("small", db=object(), whisper=whisper) == "small"


def test_empty_string_counts_as_omitted():
    assert transcribe.resolve_transcription_model("", db=object(), whisper=_FakeWhisper(cached={"large"})) == "large"


def test_configured_model_is_used_when_downloaded():
    whisper = _FakeWhisper(loaded_size="base", cached={"large"})

    assert transcribe.resolve_transcription_model(None, db=object(), whisper=whisper) == "large"


def test_configured_model_is_used_when_already_loaded():
    whisper = _FakeWhisper(loaded_size="large", cached=())

    assert transcribe.resolve_transcription_model(None, db=object(), whisper=whisper) == "large"


def test_uncached_configured_model_falls_back_to_loaded_size_with_warning(caplog):
    whisper = _FakeWhisper(loaded_size="base", cached=())

    with caplog.at_level(logging.WARNING):
        size = transcribe.resolve_transcription_model(None, db=object(), whisper=whisper)

    assert size == "base"
    assert any("not downloaded" in r.getMessage() for r in caplog.records)


def test_without_backend_the_setting_is_returned():
    assert transcribe.resolve_transcription_model(None, db=object()) == "large"


# MCP voicebox.transcribe helper


class _FakeDB:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def mcp_env(monkeypatch, tmp_path):
    import numpy as np

    import backend.utils.audio as audio_utils

    whisper = _FakeWhisper(loaded_size="base", cached={"large"})
    db = _FakeDB()
    calls = {"get_db": 0}

    def fake_get_db():
        calls["get_db"] += 1
        yield db

    monkeypatch.setattr(transcribe, "get_whisper_model", lambda: whisper)
    monkeypatch.setattr(tools, "get_db", fake_get_db)
    monkeypatch.setattr(audio_utils, "load_audio", lambda path: (np.zeros(2400, dtype="float32"), 24000))
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    return whisper, db, calls, wav


async def test_mcp_explicit_model_skips_the_settings_lookup(mcp_env):
    whisper, _db, calls, wav = mcp_env

    result = await tools._transcribe_file(wav, "es", "base")

    assert result["model"] == "base"
    assert whisper.calls == ["base"]
    assert calls["get_db"] == 0


async def test_mcp_omitted_model_follows_the_setting_and_closes_the_session(mcp_env):
    whisper, db, calls, wav = mcp_env

    result = await tools._transcribe_file(wav, "es", None)

    assert result["model"] == "large"
    assert whisper.calls == ["large"]
    assert calls["get_db"] == 1
    assert db.closed is True
