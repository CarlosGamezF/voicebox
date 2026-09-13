"""The default Whisper size follows the persisted STT setting everywhere.

``POST /transcribe`` (used by the voice-sample "Transcribe" button, REST
callers and the MCP transcribe tool) used to fall back to whatever size the
STT backend happened to hold, i.e. ``base`` on a cold backend, even when the
user had configured a larger model in Settings. The resolver below makes the
configured ``stt_model`` the default for every path; an explicit request
still wins.
"""

from backend.services import transcribe


class _Settings:
    stt_model = "large"


def test_explicit_model_wins(monkeypatch):
    monkeypatch.setattr(transcribe, "get_capture_settings", lambda db: _Settings())

    assert transcribe.resolve_transcription_model("small", db=object()) == "small"


def test_omitted_model_follows_the_persisted_stt_setting(monkeypatch):
    monkeypatch.setattr(transcribe, "get_capture_settings", lambda db: _Settings())

    assert transcribe.resolve_transcription_model(None, db=object()) == "large"


def test_empty_string_counts_as_omitted(monkeypatch):
    monkeypatch.setattr(transcribe, "get_capture_settings", lambda db: _Settings())

    assert transcribe.resolve_transcription_model("", db=object()) == "large"
